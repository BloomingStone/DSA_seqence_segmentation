from pathlib import Path
import os
import tomllib

import onnx
import numpy as np
from numpy.testing import assert_allclose
import onnxruntime
import torch
from torch.export import Dim
from torch import nn
from monai.transforms.post.array import AsDiscrete
from monai.metrics.meandice import DiceMetric

from ...core.utils.task_type import TaskType
from .image_type import ImageType
from ...core.base_models.ResEncUNet_pure_torch import ResEncUNet
from .dataloader import DataLoaderManager


def export_onnx(
    task_dir: Path = Path(__file__).parent,
    model: ResEncUNet = ResEncUNet(output_channels=1),
    params: dict | None = None
):
    if params is None:
        with open(task_dir / "params.toml", "rb") as f:
            params = tomllib.load(f)
    assert params is not None

    checkpoint_dir = task_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = params["device"]["CUDA_VISIBLE_DEVICES"]
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    model = model.to('cuda')

    checkpoint_path = task_dir / "checkpoints" /" best.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda', weights_only=True))

    dataloader_manager = DataLoaderManager(
        train_image_dir=Path(params["dataset"]["train_images_dir"]),
        train_label_dir=Path(params["dataset"]["train_labels_dir"]),
        test_image_dir=Path(params["dataset"]["test_images_dir"]),
        test_label_dir=Path(params["dataset"]["test_labels_dir"]),
        sample_num=10
    )

    loader = dataloader_manager.get_dataloader(
        TaskType.Valid,
        batch_size=2,
        shuffle=False
    )

    model.half()
    model.eval()
    
    dummy_input = torch.randn(1, 1, 512, 512).to('cuda').half()
    onnx_path = checkpoint_dir / 'ResEncUNet.onnx'
    torch.onnx.export(
        model,
        (dummy_input, ),
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # 获取图结构
    graph = onnx_model.graph

    # 查看输入
    print("Inputs:")
    for input_tensor in graph.input:
        print(f"  Name: {input_tensor.name}")
        tensor_type = input_tensor.type.tensor_type
        shape = [dim.dim_value for dim in tensor_type.shape.dim]
        print(f"  Shape: {shape}")
        print(f"  Data type: {tensor_type.elem_type}")  # 对应onnx.TensorProto中的类型编号

    # 查看输出
    print("Outputs:")
    for output_tensor in graph.output:
        print(f"  Name: {output_tensor.name}")
        tensor_type = output_tensor.type.tensor_type
        shape = [dim.dim_value for dim in tensor_type.shape.dim]
        print(f"  Shape: {shape}")
        print(f"  Data type: {tensor_type.elem_type}")

    ort_session = onnxruntime.InferenceSession(
        str(onnx_path), providers=['CUDAExecutionProvider']
    )

    dice_metric = DiceMetric(include_background=False)
    as_discrete = AsDiscrete(threshold=0.5)

    test_input = next(iter(loader))[ImageType.Image].as_tensor().to('cuda').half()
    test_label = next(iter(loader))[ImageType.Label].as_tensor().to('cuda').half()
    torch_output = model(test_input)
    torch_output = as_discrete(torch_output)
    assert isinstance(torch_output, torch.Tensor)
    torch_dice = dice_metric(torch_output, test_label)
    assert isinstance(torch_dice, torch.Tensor)
    print(torch_dice.mean().item())

    input_np = test_input.detach().cpu().numpy()
    input_ort_value = onnxruntime.OrtValue.ortvalue_from_numpy(input_np)
    io_binding = ort_session.io_binding()
    io_binding.bind_input(
        name='input',
        device_type=input_ort_value.device_name(),
        device_id=0,
        element_type=input_np.dtype,
        shape=input_np.shape,
        buffer_ptr=input_ort_value.data_ptr()
    )
    io_binding.bind_output(
        name='output',
        device_type=input_ort_value.device_name(),
        device_id=0,
        element_type=input_np.dtype,
        shape=input_np.shape
    )
    ort_session.run_with_iobinding(io_binding)
    onnx_output = io_binding.get_outputs()[0].numpy()
    onnx_output = as_discrete(onnx_output)

    assert isinstance(onnx_output, torch.Tensor)
    dice_onnx = dice_metric(
        onnx_output.to(device='cuda', dtype=test_label.dtype), 
        test_label
    )
    assert isinstance(dice_onnx, torch.Tensor)
    print(dice_onnx.mean().item())
    

    try:
        np.testing.assert_allclose(
            torch_output.detach().cpu().numpy(),
            onnx_output.detach().cpu().numpy(),
            verbose=True,
            rtol=1e-02,
        )
    except AssertionError as e:
        print(e)


if __name__ == '__main__':
    export_onnx()