## CNN Model for CIFAR-10 dataset

# Importing required libraries
import time

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import cuda, device, float32, nn, no_grad, optim, randint, save, utils
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


# Create CNN Model
class CnnModule(nn.Module):
    def __init__(self, num_output):
        super().__init__()
        self.cnn_model_1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, "same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, 1, "same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, 1, "same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
        )

        self.cnn_model_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, "same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1, "same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, "same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )

        self.cnn_model_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, "same"),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 1, "same"),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 1, "same"),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        # Image Flatten Caluclation (256*4*4)
        self.fcn_model = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(4 * 4 * 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_output),
        )

    def forward(self, input_batch):
        out = self.cnn_model_1(input_batch)
        out = self.cnn_model_2(out)
        out = self.cnn_model_3(out)
        out = out.view(out.size(0), -1)
        out = self.fcn_model(out)

        return out


# Training loop function
def run_training_loop(trainSet_loader, testSet_loader, num_epoch=10, batch_size=64):
    training_losses = []
    testing_losses = []

    # Instantiating the Model and its Parameters
    model = CnnModule(num_output=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Tranferring the model to CUDA
    device_name = device("cuda:0" if cuda.is_available() else "cpu")
    print(device_name)
    model.to(device_name)

    ## ------------ RUNNING EPOCHS ----------------- ##
    for epoch in range(1, num_epoch + 1):
        start = time.time()
        # Training Loop per epoch
        model.train()
        batchTrain_loss = 0
        count = 0
        for images, labels in trainSet_loader:
            images, labels = images.to(device_name), labels.to(device_name)
            optimizer.zero_grad()

            pred_labels = model(images)
            loss = criterion(pred_labels, labels)

            batchTrain_loss += loss.item()
            count += 1

            loss.backward()
            optimizer.step()

        if count:
            training_losses.append(batchTrain_loss / count)

        # Testing Loop per epoch
        model.eval()
        batchTest_loss = 0
        count = 0
        with no_grad():
            for images, labels in testSet_loader:
                images, labels = images.to(device_name), labels.to(device_name)
                pred_labels = model(images)
                loss = criterion(pred_labels, labels)

                batchTest_loss += loss.item()
                count += 1

            if count:
                testing_losses.append(batchTest_loss / count)

        # Print the results
        duration = time.time() - start
        print(
            f"[{duration:.2f} sec]Epoch-{epoch} \
            >> Train Loss: {training_losses[-1]} \
            >> Test Loss: {testing_losses[-1]}"
        )

    # Return the losses and Trained Weights and Biases
    return model, training_losses, testing_losses


# Confusion Matrix Plotting Function
def plot_the_confusion_matrix(model, testSet_loader):
    device_name = device("cuda:0" if cuda.is_available() else "cpu")
    model.to(device_name)

    all_preds = []
    all_labels = []
    with no_grad():
        for images, labels in testSet_loader:
            images, labels = images.to(device_name), labels.to(device_name)
            pred_labels = model(images)
            _, preds = nn.functional.softmax(pred_labels, dim=1).max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


# Miscalssified Data Plotting Function
def plot_misclassified_data(model, testSet_loader):
    # label mapping
    txt_label = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }
    device_name = device("cuda:0" if cuda.is_available() else "cpu")
    model.to(device_name)
    with no_grad():
        # random number between 0 and 20
        idx = randint(0, 20, (1,)).item()
        for _ in range(idx):
            images, labels = next(iter(testSet_loader))

        images, labels = images.to(device_name), labels.to(device_name)
        pred_labels = model(images)
        _, preds = nn.functional.softmax(pred_labels, dim=1).max(1)
        missclassified_idx = (preds != labels).nonzero()

    # Display a 10 random training image in a grid
    images, labels, preds = images.cpu(), labels.cpu(), preds.cpu()

    plt.figure(figsize=(10, 10))
    for i in range(len(missclassified_idx)):
        index = missclassified_idx[i].item()
        image, label = images[index], labels[index].item()
        # convert the image to have a range of [0, 1]
        image = image - image.min()
        image = image / image.max()

        plt.subplot(5, 5, i + 1)
        plt.imshow(image.squeeze().moveaxis(0, -1), cmap="gray")
        plt.title(f"{txt_label[label]}|{txt_label[preds[index].item()]}")
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Downloading CIFAR-10 dataset
    transform_func = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(float32),
            # v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(10),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root="data/", train=True, transform=transform_func, download=True
    )
    test_dataset = datasets.CIFAR10(
        root="data/", train=False, transform=transform_func, download=True
    )

    # Getting the dataset attributes
    print(f"Training Sample : {len(train_dataset)}")
    print(f"Testing Sample : {len(test_dataset)}")
    print(f"No. of Label: {len(train_dataset.classes)}")
    print(f"Image Shape : {train_dataset.data.shape[1:]}")

    # Instantiating DataLoader
    trainSet_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    testSet_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

    # Training the model
    trained_model, training_losses, testing_losses = run_training_loop(
        trainSet_loader, testSet_loader, 20, 512
    )

    # Saving the trained model
    save(trained_model.state_dict(), "models/model_cifar10_v1.pt")

    # Plotting the Losses
    plt.plot(training_losses, "-bx", label="Training Loss")
    plt.plot(testing_losses, "-rx", label="Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plotting the Confusion Matrix
    plot_the_confusion_matrix(trained_model, testSet_loader)

    # Plot some misscalssified images
    plot_misclassified_data(trained_model, testSet_loader)
