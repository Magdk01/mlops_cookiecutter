import click
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data.mnist_dataloader import mnist
from models.model import MyAwesomeModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--e", default=5, help="Epochs for training")
def train(lr, e):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_set, _ = mnist()
    # train_set.to(device)
    loss_list = []
    for epoch in range(e):
        print(epoch)
        for batch in tqdm(train_set):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        loss_list.append(loss.item())
    torch.save(model, "models/trained_models/trained_model_v1.pt")

    plt.plot(loss_list)
    plt.title("Loss per epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig("reports/figures/loss_curve.png")


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    model.to(device)
    model.eval()
    _, test_set = mnist(64)
    correct = 0
    total = 0
    with torch.no_grad():
        for target in test_set:
            inputs, labels = target
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
