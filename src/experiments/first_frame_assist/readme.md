# DSA 导丝引导线分割模型

其中 `train.py` 是训练模型的代码, `params.toml`是一些训练参数, `only_test.py` 只运行valid_epoch代码, `predict.py` 可以直接预测整个序列, `/checpoints/` 下是训练好的模型权重.

params.toml 中的`basemodel_checkpoint_path`是之前训练好的单帧分割模型,其中的`encoder` 和`decoder`模块和现在的模型基本保持一致.

`/export_onnx/`下是导出的onnx模型, 一共有两个onnx文件, 分别是 `fisrt_frame_encoder.onnx` 和 `mix_and_decoder.onnx`, 前者用来提取第一帧特征, 后者用来提取后续帧特征并于第一帧特征混合后解码得到分割结果. 以python代码所写的运行示例见`export_onnx.py` 的 `test_onnx` 函数.

其各自的输入输出如下:

``` json
{
    "first_frame_encoder.onnx":{
        "input": {
            [
                {
                    "name": "first_frame",
                    "shape": [1, 1, 512, 512]
                }
            ]
        },
        "output": {
            [
                {
                    "name": "feature_0" ,
                    "shape": [1, 32, 512, 512]
                },
                {
                    "name": "feature_1",
                    "shape": [1, 64, 256, 256]
                },
                {
                    "name": "feature_2",
                    "shape": [1, 128, 128, 128]
                },
                {
                    "name": "feature_3",
                    "shape": [1, 256, 64, 64]
                },
                {
                    "name": "feature_4",
                    "shape": [1, 512, 32, 32]
                },
                {
                    "name": "feature_5",
                    "shape": [1, 512, 16, 16]
                },
                {
                    "name": "feature_6",
                    "shape": [1, 512, 8, 8]
                },
                {
                    "name": "feature_7",
                    "shape": [1, 512, 4, 4]
                }
            ]
        }
    },
    "mix_and_decoder.onnx":{
        "input": {
            [
                {
                    "name": "frame_n",
                    "shape": ["batch", 1, 512, 512]
                },
                {
                    "name": "feature_0" ,
                    "shape": [1, 32, 512, 512]
                },
                {
                    "name": "feature_1",
                    "shape": [1, 64, 256, 256]
                },
                {
                    "name": "feature_2",
                    "shape": [1, 128, 128, 128]
                },
                {
                    "name": "feature_3",
                    "shape": [1, 256, 64, 64]
                },
                {
                    "name": "feature_4",
                    "shape": [1, 512, 32, 32]
                },
                {
                    "name": "feature_5",
                    "shape": [1, 512, 16, 16]
                },
                {
                    "name": "feature_6",
                    "shape": [1, 512, 8, 8]
                },
                {
                    "name": "feature_7",
                    "shape": [1, 512, 4, 4]
                }
            ]
        },
        "output": {
            [
                {
                    "name": "output",
                    "shape": ["batch", 1, 512, 512]
                }
            ]
        }
    }
}
```
