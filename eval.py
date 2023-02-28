import torch
import argparse
from model import DDP

def evaluate(model, dataloader, device):
    """Evaluate the performance of the model on a given dataset.

    Args:
        model (nn.Module): Trained model to be evaluated.
        dataloader (DataLoader): Dataloader for the dataset to be evaluated.
        device (torch.device): Device on which to run the evaluation.
    Returns:
        float: Average loss over the dataset.
    """
    model.eval()
    total_loss = 0.
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for data in dataloader:
            x = data.to(device)
            output = model(x)
            loss = criterion(output, x)
            total_loss += loss.item() * x.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the DDP model.')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to dataset directory.')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model file.')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for evaluation.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation.')
    args = parser.parse_args()

    # Load the dataset and dataloader
    dataset = torch.load(args.data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)

    # Load the trained model
    device = torch.device(args.device)
    model = DDP(input_dim=dataset.shape[1], output_dim=dataset.shape[1], num_diffusion_steps=10)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # Evaluate the model on the dataset
    avg_loss = evaluate(model, dataloader, device)
    print(f'Average loss: {avg_loss:.4f}')
