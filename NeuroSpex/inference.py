import argparse
import torch
import json
from Models.neurospex import NeuroSpex
from utils import SISDRLoss, SISNRLoss, custom_collate_fn
from tqdm import tqdm
from Data.dataset import NeurHeedEEG_Dataset
from torch.utils.data import DataLoader 


class NeuroSpexInference:
    def __init__(
        self, 
        model_config, 
        checkpoint_path, 
        gpu=None, 
        savedir=None
    ):
        self.config = model_config
        self.savedir = savedir
        # Initialize model
        self.model = NeuroSpex(
            self.config['speech_encoder_params'],
            self.config['eeg_encoder_params'],
            self.config['speech_decoder_params'],
            self.config['spk_ext_params']
        )
        self.gpu = gpu
        self.metric = SISNRLoss()

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            #self._load_model(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
            if self.gpu is not None:
                self.model = self.model.to(self.gpu)
        else:
            print(f"No checkpoint found at {checkpoint_path}")
    
        self.SISDR = []

    # def _load_model(self, path, load_optimizer=False, load_training_stat=False):
    #     checkpoint = torch.load(path, map_location='cpu')
    #     # load model weights
    #     #pretrained_model = checkpoint['model']
    #     #state = self.model.state_dict()
    #     #for key in pretrained_model.keys():
    #     #    state_key = ".".join(key.split('.')[2:])
    #     #    state[state_key] = pretrained_model[key]
    #     self.model.load_state_dict( checkpoint['model_state_dict'])
        

    def infer(self, mixture, eeg, target):
        self.model.eval()
        with torch.no_grad():
            if self.gpu is not None:
                mixture = mixture.to(self.gpu)
                eeg = eeg.to(self.gpu)
                target = target.to(self.gpu)
            estimated = self.model(mixture, eeg, speech_query=self.config["speech_query"])
            sisdr = -self.metric(estimated.squeeze(1), target.squeeze(1))
            self.SISDR.append(sisdr)
        return sisdr

    def run(self, test_loader):
        total_sisdr = 0.0
        num_batches = 0

        for batch in tqdm(test_loader):
            mixture, target, eeg = batch
            sisdr = self.infer(mixture, eeg, target)
            total_sisdr += sisdr.item()
            num_batches += 1

        avg_sisdr = total_sisdr / num_batches if num_batches > 0 else 0.0
        print(f"Average SI-SDR over test set: {avg_sisdr:.4f} dB")

        #Write output file
        save_dir = f"{self.save_dir}/{self.config["run_name"]}"
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/test_score_{avg_sisdr:.4f}.out", "w") as f:
            for i, sisdr in enumerate(self.SISDR):
                f.write(f"{i} {sisdr:.4f}\n")
        return avg_sisdr


def main(ARGS):

    # Load config
    with open(ARGS.config, 'r') as f:
        config = json.load(f)

    test_dataset = NeurHeedEEG_Dataset(
        root=config["data"]["root"], 
        partition='test',
        batch_size=1, 
        max_length=10, 
        audio_sr=8000, 
        ref_sr=128)

    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,
        shuffle=False, 
        num_workers=config["train"]["batch_size"],
        collate_fn=custom_collate_fn
    )

    pipeline = NeuroHeedInference(
        model_config=config["train"]["neuroheed_params"],
        checkpoint_path=ARGS.checkpoint,
        gpu=0,
        save_dir=ARGS.savedir
    )

    pipeline.run(test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('-pt', '--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('-o', '--output', type=str, rewuired=True, help='Path to directory where outputs are stored')
    args = parser.parse_args()
    main(args)