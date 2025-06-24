from pathlib import Path
import os

import torch
from monai.data import ThreadDataLoader, MetaTensor, CacheDataset
from monai import transforms as T
from torch import nn
import nibabel as nib
import numpy as np
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

from .model import ResEncUNet_FirstFrameAssist


Sequence = 'sequence'

def get_sequence_transform() -> T.Compose:
    return T.Compose([
        T.LoadImaged(keys=[Sequence], image_only=True),
        T.EnsureChannelFirstd(keys=[Sequence]),
        T.EnsureTyped(keys=[Sequence], dtype=torch.float32),
        T.ClipIntensityPercentilesd(keys=[Sequence], lower=1, upper=99, sharpness_factor=5),
        T.NormalizeIntensityd(keys=[Sequence], nonzero=True),
    ])
    

def get_sequence_predict_dataloader(nii_image_dir: Path):
    print("loading data")
    nii_image_paths = list(sorted(nii_image_dir.glob('*.nii.gz')))
    data = [{
        Sequence: nii_image_path,
    } for nii_image_path in nii_image_paths]
    dataset = CacheDataset(
        data,
        transform=get_sequence_transform(),
    )
    return ThreadDataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

# TODO 这个可以放在core里
def save_mp4(ori_sequence: np.ndarray, mask_sequence: np.ndarray, output_path: Path, fps=10):
    frames = []
    for i in tqdm(range(ori_sequence.shape[0]), desc="output video"):
        fig, ax = plt.subplots(figsize=(4,4), dpi=128)
        ax.imshow(ori_sequence[i], cmap='gray')
        ax.imshow(mask_sequence[i], alpha=0.4, cmap='Reds')
        plt.axis('off')
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(image)
        plt.close(fig)
    writer = imageio.get_writer(output_path, fps=fps)
    for f in frames:
        writer.append_data(f)
    writer.close()

class Predictor:
    def __init__(self, model: nn.Module, checkpoint_path: Path):
        self.model = model.to('cuda')
        self.checkpoint_path = checkpoint_path
        self.model.load_state_dict(torch.load(
            checkpoint_path,
            map_location='cuda',
            weights_only=True,
        ))
        self.model.eval()

    def predict_sequence(self, sequence: MetaTensor) -> MetaTensor:
        """
        预测整个序列
        :param sequence: 输入序列, shape = (B=1, C=1, T, W, H)
        :return: 输出预测mask 序列, shape = (B=1, C=1, T, W, H)
        """
        as_discrete = T.AsDiscrete(threshold=0.5)
        first_frame = sequence[:, :, 0, :, :]
        n_frames = sequence.shape[2]
        result = torch.zeros_like(sequence)
        for i in tqdm(range(n_frames), desc="segmentation"):
            x = sequence[:, :, i, :, :]
            with torch.no_grad():
                logit = self.model(x, first_frame)
            predict = as_discrete(logit)
            result[:, :, i, :, :] = predict
        return result

    def predict(
            self,
            image_dir: Path,
            output_mask_nii_dir: Path,
            output_mp4_dir: Path
    ):
        assert image_dir.exists()
        output_mask_nii_dir.mkdir(exist_ok=True, parents=True)
        output_mp4_dir.mkdir(exist_ok=True, parents=True)
        dataloader = get_sequence_predict_dataloader(image_dir)
        length = len(dataloader)
        for i, batch_data in enumerate(dataloader):
            sequence = batch_data[Sequence].cuda()
            sequence_name = Path(sequence.meta['filename_or_obj'][0]).stem.split('.')[0]

            print(f"predicting {sequence_name} [ {i+1} / {length} ]")
            predict_label = self.predict_sequence(sequence).squeeze().cpu().numpy()

            # save mask as nii.gz
            output_nii_path = output_mask_nii_dir / f'{sequence_name}.nii.gz'
            label_image = nib.Nifti1Image(predict_label, affine=np.eye(4)) 
            nib.save(label_image, output_nii_path)

            # save image+mask overlay as mp4
            output_mp4_path = output_mp4_dir / f'{sequence_name}.mp4'
            original_sequence = sequence.squeeze().cpu().numpy() # T, W, H
            save_mp4(original_sequence, predict_label, output_mp4_path)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = ResEncUNet_FirstFrameAssist(num_classes = 1).cuda()
    best_ckpt_path = Path(__file__).parent / 'checkpoints copy' / 'best.pth'
    predictor = Predictor(model, best_ckpt_path)
    predictor.predict(
        image_dir = Path("/media/data3/sj/Data/dsa/2024/L_ori_nii"),
        output_mask_nii_dir= Path("/media/data3/sj/Data/dsa/2024/L_ori_predict_nii"),
        output_mp4_dir=Path("/media/data3/sj/Data/dsa/2024/L_ori_video")
    )

if __name__ == "__main__":
    main()