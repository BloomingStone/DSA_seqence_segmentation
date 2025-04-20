from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import onnx
import onnxruntime
import torch
from torch import nn

from .model import ResEncUNet_FirstFrameAssist

class MixAndDecoder(nn.Module):
    def __init__(self, model: ResEncUNet_FirstFrameAssist):
        super().__init__()
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.feature_mixtures = model.feature_mixtures
    
    def forward(
            self, 
            x,
            *skips_first_frame
    ) -> torch.Tensor:
        skips = self.encoder(x)
        for i in range(len(skips)):
            skips[i] = self.feature_mixtures[i](
                torch.cat([skips[i], skips_first_frame[i]], dim=1)
            )

        del skips_first_frame

        out = self.decoder(skips)
        if self.training:
            res = []
            output_shape = out[0].shape
            res.append(out[0])
            for i in range(1, len(out)):
                res.append(nn.functional.interpolate(out[i], output_shape[2:], mode='bilinear', align_corners=True))
            return res
        else:
            return out[0]
        


def export_onnxs(
        model: ResEncUNet_FirstFrameAssist, 
        output_dir: Path
    ) -> tuple[list[str], list[tuple[int]]]:
    output_dir.mkdir(exist_ok=True, parents=True)
    model.eval()
    first_frame_encoder = model.encoder_first_frame
    features_per_stage = model.features_per_stage
    strides = model.strides
    devide_by = np.cumprod(strides)
    size = [512 // d for d in devide_by]
    features = [torch.randn(1, f, s, s) for f, s in zip(features_per_stage, size)]
    features_name = ['feature_{i}' for i in range(len(features_per_stage))]

    torch.onnx.export(
        first_frame_encoder,
        torch.randn(1, 1, 512, 512),
        output_dir / 'first_frame_encoder.onnx',
        input_names=['first_frame'],
        output_names=features_name,
    )

    mix_and_decoder = MixAndDecoder(model)
    mix_and_decoder.eval()
    torch.onnx.export(
        mix_and_decoder,
        (torch.randn(1, 1, 512, 512), *features),
        output_dir / 'mix_and_decoder.onnx',
        input_names=['frame_n', *features_name],
        output_names=['output'],
    )
    return (
        features_name,
        [f.shape for f in features]
    )

def test_onnx(
        first_frame_onnx_file: Path,
        mix_and_decoder_onnx_file: Path,
        first_frame_numpy: np.ndarray,
        frame_n_numpy: np.ndarray,
        freatures_name: list[str],
        freatures_shape: list[tuple[int]]
) -> np.ndarray:
    first_frame_onnx = onnx.load(str(first_frame_onnx_file))
    onnx.checker.check_model(first_frame_onnx)
    first_frame_session = onnxruntime.InferenceSession(
        str(first_frame_onnx_file),
        providers=['CUDAExecutionProvider']
    )

    mix_and_decoder_onnx = onnx.load(str(mix_and_decoder_onnx_file))
    onnx.checker.check_model(mix_and_decoder_onnx)
    mix_and_decoder_session = onnxruntime.InferenceSession(
        str(mix_and_decoder_onnx_file),
        providers=['CUDAExecutionProvider']
    )

    first_frame_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
        first_frame_numpy, "cuda", 0
    )

    frame_n_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
        frame_n_numpy, "cuda", 0
    )

    first_frame_io_binding = first_frame_session.io_binding()
    first_frame_io_binding.bind_input(
        name='first_frame', device_type="cuda", device_id=0, 
        element_type=np.float32, shape=first_frame_numpy.shape, 
        buffer_ptr=first_frame_ortvalue.data_ptr()
    )
    for name, feature_shape in zip(freatures_name, freatures_shape):
        first_frame_io_binding.bind_output(
            name=name, device_type='cuda', device_id=0, 
            element_type=np.float32, shape=feature_shape
        )
    
    first_frame_session.run_with_iobinding(first_frame_io_binding)

    features = [
        first_frame_io_binding.get_outputs()[i].numpy()
        for i in range(len(freatures_name))
    ]

    mix_and_decoder_io_binding = mix_and_decoder_session.io_binding()
    mix_and_decoder_io_binding.bind_input(
        name='frame_n', device_type="cuda", device_id=0, 
        element_type=np.float32, shape=frame_n_numpy.shape, 
        buffer_ptr=frame_n_ortvalue.data_ptr()
    )
    for i, feature in enumerate(features):
        mix_and_decoder_io_binding.bind_input(
            name=freatures_name[i], device_type='cuda', device_id=0, 
            element_type=np.float32, shape=feature.shape, 
            buffer_ptr=feature.data_ptr()
        )
    mix_and_decoder_io_binding.bind_output(
        name='output', device_type='cuda', device_id=0, 
        element_type=np.float32, shape=frame_n_numpy.shape
    )
    mix_and_decoder_session.run_with_iobinding(mix_and_decoder_io_binding)
    return mix_and_decoder_io_binding.get_outputs()[0].numpy()


def test():
    model = ResEncUNet_FirstFrameAssist(num_classes=1)
    model.load_state_dict(
        torch.load('model.pth', map_location='cuda', weights_only=True), 
        strict=False
    )
    model.cuda()
    model.eval()
    output_dir = Path('onnx_export')
    features_name, features_shape = export_onnxs(model, output_dir)
    print(f"features_name: {features_name}, features_shape: {features_shape}")
    first_frame_numpy = np.random.randn(1, 1, 512, 512).astype(np.float32)
    frame_n_numpy = np.random.randn(1, 1, 512, 512).astype(np.float32)
    onnx_output = test_onnx(
        first_frame_onnx_file=output_dir / 'first_frame_encoder.onnx',
        mix_and_decoder_onnx_file=output_dir / 'mix_and_decoder.onnx',
        first_frame_numpy=first_frame_numpy,
        frame_n_numpy=frame_n_numpy,
        freatures_name=features_name,
        freatures_shape=features_shape
    )
    torch_output = model(
        torch.from_numpy(first_frame_numpy).cuda(),
        torch.from_numpy(frame_n_numpy).cuda()
    )
    torch_output = torch_output[0].cpu().detach().numpy()

    assert_allclose(
        onnx_output, torch_output, rtol=1e-03, atol=1e-05
    )
    

if __name__ == '__main__':
    test()    