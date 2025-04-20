from pathlib import Path
import os

import numpy as np
from numpy.testing import assert_allclose
import onnxruntime
import torch
from torch.export import Dim
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
            frame_n,
            *skips_first_frame
    ) -> torch.Tensor:
        skips = self.encoder(frame_n)
        for i in range(len(skips)):
            skips[i] = self.feature_mixtures[i](
                torch.cat([
                        skips[i], 
                        skips_first_frame[i].repeat((frame_n.shape[0], 1, 1, 1))
                    ], 
                    dim=1
                )
            )

        del skips_first_frame
        return self.decoder(skips)[0]


def export_onnxs(
        model: ResEncUNet_FirstFrameAssist, 
        output_dir: Path
    ) -> tuple[list[str], list[tuple[int]]]:
    output_dir.mkdir(exist_ok=True, parents=True)
    model.eval()
    first_frame_encoder = model.encoder_first_frame
    first_frame_encoder.eval()
    features_per_stage = model.features_per_stage
    strides = model.strides
    divide_by = np.cumprod(strides)
    size = [512 // d for d in divide_by]
    features = [torch.randn(1, f, s, s).cuda() for f, s in zip(features_per_stage, size)]
    features_name = [f'feature_{i}' for i in range(len(features_per_stage))]

    torch.onnx.export(
        first_frame_encoder,    
        torch.randn(1, 1, 512, 512).cuda(), # 输入：第一帧，shape固定为 (1, 1, 512, 512)
        output_dir / 'first_frame_encoder.onnx',
        input_names=['first_frame'],    
        output_names=features_name,     # 输出名：['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7']
    )                                   # 输出：第一帧提取特征，shape 依次为[1, 32, 512, 512], [1, 64, 256, 256], [1, 256, 64, 64], [1, 512, 32, 32], [1, 512, 16, 16], [1, 512, 8, 8], [1, 512, 4, 4]

    mix_and_decoder = MixAndDecoder(model)
    mix_and_decoder.eval()
    inputs = (torch.randn(1, 1, 512, 512).cuda(), *features)
    torch.onnx.export(
        mix_and_decoder,
        inputs,
        output_dir / 'mix_and_decoder.onnx',
        input_names=['frame_n', *features_name],    # 输入：第n帧和第一帧特征, 其中，第n帧的shape为（B, 1, 512, 512）, 既可以将多帧一起输入
        output_names=['output'],                    # 输出：第n帧分割结果的logits, shape为(B, 1, 512, 512), 同上
        dynamic_axes={
            'frame_n': {0: 'batch'},
            'output': {0: 'batch'}
        }
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
        features_name: list[str],
        features_shape: list[tuple[int]],
        device_id: int = 1
) -> np.ndarray:
    def get_session(onnx_file: Path) -> onnxruntime.InferenceSession:
        return onnxruntime.InferenceSession(
            str(onnx_file),
            providers=[('CUDAExecutionProvider', {'device_id': device_id})]
        )
    
    def get_ort_value(x: np.array) -> onnxruntime.OrtValue:
        return onnxruntime.OrtValue.ortvalue_from_numpy(x, "cuda", device_id)
    
    def bind(
        session_io: onnxruntime.IOBinding,
        name: str,
        shape: tuple,
        ort_value: onnxruntime.OrtValue | None = None,
    ):
        if ort_value is not None:
            session_io.bind_input(
                name=name, device_type='cuda', device_id=device_id,
                element_type=np.float32, shape=shape, buffer_ptr=ort_value.data_ptr()
            )
        else:
            session_io.bind_output(
                name=name, device_type='cuda', device_id=device_id, 
                element_type=np.float32, shape=shape
            )


    first_frame_session = get_session(first_frame_onnx_file)
    mix_and_decoder_session = get_session(mix_and_decoder_onnx_file)

    first_frame_ort_value = get_ort_value(first_frame_numpy)
    frame_n_ort_value = get_ort_value(frame_n_numpy)

    first_frame_io = first_frame_session.io_binding()
    bind(first_frame_io, 'first_frame', first_frame_numpy.shape, first_frame_ort_value)
    for name, shape in zip(features_name, features_shape):
        bind(first_frame_io, name, shape)
    first_frame_session.run_with_iobinding(first_frame_io)
    features: list[np.ndarray] = [ first_frame_io.get_outputs()[i].numpy() for i in range(len(features_name)) ]

    mix_and_decoder_io = mix_and_decoder_session.io_binding()
    bind(mix_and_decoder_io, 'frame_n', frame_n_numpy.shape, frame_n_ort_value)
    for name, feature, shape in zip(features_name, features, features_shape):
        feature_ort_value = onnxruntime.OrtValue.ortvalue_from_numpy(feature, "cuda", device_id)
        bind(mix_and_decoder_io, name, shape, feature_ort_value)
    bind(mix_and_decoder_io, 'output', frame_n_numpy.shape)
    mix_and_decoder_session.run_with_iobinding(mix_and_decoder_io)
    return mix_and_decoder_io.get_outputs()[0].numpy()


def test():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    model = ResEncUNet_FirstFrameAssist(num_classes=1)
    model.load_state_dict(
        torch.load('/media/data3/sj/Code/DSA/src/experiments/first_frame_assist/checkpoints/best.pth', map_location='cuda', weights_only=True), 
        strict=False
    )
    model.eval()
    model.cuda()
    first_frame_numpy = np.random.randn(1, 1, 512, 512).astype(np.float32)
    frame_n_numpy = np.random.randn(3, 1, 512, 512).astype(np.float32)
    torch_output = model(
        torch.from_numpy(frame_n_numpy).cuda(),
        torch.from_numpy(first_frame_numpy).repeat((3, 1, 1, 1)).cuda(),
    )
    torch_output = torch_output.cpu().detach().numpy()

    output_dir = Path(__file__).parent / 'exported_onnx'
    features_name, features_shape = export_onnxs(model, output_dir)

    print(f"features_name: {features_name}, \nfeatures_shape: {features_shape}")
    onnx_output = test_onnx(
        first_frame_onnx_file=output_dir / 'first_frame_encoder.onnx',
        mix_and_decoder_onnx_file=output_dir / 'mix_and_decoder.onnx',
        first_frame_numpy=first_frame_numpy,
        frame_n_numpy=frame_n_numpy,
        features_name=features_name,
        features_shape=features_shape
    )

    assert_allclose(
        onnx_output, torch_output, rtol=1e-03, atol=1e-03
    )
    

if __name__ == '__main__':
    test()    