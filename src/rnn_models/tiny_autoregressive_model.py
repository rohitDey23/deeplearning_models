import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

np.random.seed(2)


class AutoregressiveModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model_construction()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

        self.training_losses = []
        self.testing_losses = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def forward(self, x):
        y = self.model(x)
        return y

    def model_construction(self):
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

        return model

    def train_model(self, input_data, target_data, num_epochs=1000):
        training_losses = []
        testing_losses = []

        # Divide the data into training and testing sets
        split = int(0.5 * len(input_data))
        x_train, y_train = (
            input_data[:split].reshape(-1, 10),
            target_data[:split].reshape(-1, 1),
        )
        x_test, y_test = (
            input_data[split:].reshape(-1, 10),
            target_data[split:].reshape(-1, 1),
        )

        # Move the data to the GPU
        x_train = torch.from_numpy(x_train.astype(np.float32)).cuda()
        y_train = torch.from_numpy(y_train.astype(np.float32)).cuda()
        x_test = torch.from_numpy(x_test.astype(np.float32)).cuda()
        y_test = torch.from_numpy(y_test.astype(np.float32)).cuda()

        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            prediction = self.model(x_train)
            loss = self.loss_fn(prediction, y_train)
            loss.backward()
            self.optimizer.step()

            training_losses.append(loss.item())

            # Evaluate the model on the test set
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(x_test)
                loss = self.loss_fn(prediction, y_test)
                testing_losses.append(loss.item())

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}, Training Loss: {training_losses[-1]}, Testing Loss: {testing_losses[-1]}"
                )

        self.training_losses = training_losses
        self.testing_losses = testing_losses

        return training_losses, testing_losses

    def predict(self, x, y, prediction_window):
        for i in range(prediction_window):
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(x)
                x = torch.cat((x[:, 1:], prediction), dim=1)
                y = torch.cat((y, prediction), dim=1)

        return y


## Helper functions
def display_data(num_lines, labels=None, *args):
    # generate random colors for each line
    colors = np.random.rand(num_lines, 3)

    for line in range(num_lines):
        if labels:
            plt.plot(
                args[line * 2],
                args[line * 2 + 1],
                color=colors[line],
                label=labels[line],
            )
        else:
            plt.plot(
                args[line * 2],
                args[line * 2 + 1],
                color=colors[line],
                label=f"Line {line + 1}",
            )

    plt.legend()
    plt.grid(True, which="major", linestyle="--", linewidth=0.5, axis="both")
    # plt.axis("equal")
    plt.show()


def create_sine_wave_data(amplitude, noise, num_points, t, ts_length):
    time_series = np.linspace(0, t, num_points)
    Ft = amplitude * np.sin(time_series) + np.random.normal(
        0, noise, size=num_points
    )  # SINE wave A=10 and with noise (0, 0.2)
    display_data(1, [], time_series, Ft)

    # Divide the data into multiple samples
    input_data = []
    target_data = []
    ts_len = ts_length

    for sample in range(len(Ft) - ts_len):
        input_data.append(Ft[sample : sample + ts_len])
        target_data.append(Ft[sample + ts_len])

    input_data = np.array(input_data)
    target_data = np.array(target_data)
    print(input_data.shape, target_data.shape)

    return input_data, target_data


if __name__ == "__main__":
    # Create and visualize simulated time series data, of a SINE wave structure
    ts_len = 10
    input_data, target_data = create_sine_wave_data(1000, 0.2, 1000, 100, ts_len)

    # Normalize the data
    mean = np.mean(np.append(input_data[:, 1], input_data[-1, 1:]).flatten())
    std = np.std(np.append(input_data[:, 1], input_data[-1, 1:]).flatten())

    input_data_norm = (input_data - mean) / std
    target_data_norm = (target_data - mean) / std

    # Create an instance of the AutoregressiveModel class
    model = AutoregressiveModel()

    # Train the model
    train_loss, test_loss = model.train_model(
        input_data_norm, target_data_norm, num_epochs=1000
    )

    display_data(
        2,
        ["training loss", "testing loss"],
        range(len(train_loss)),
        train_loss,
        range(len(test_loss)),
        test_loss,
    )

    # Predict the future values with something else they are trained on
    input_data, target_data = create_sine_wave_data(1, 0.2, 1000, 100, 10)
    # Normalize the data
    mean = np.mean(np.append(input_data[:, 1], input_data[-1, 1:]).flatten())
    std = np.std(np.append(input_data[:, 1], input_data[-1, 1:]).flatten())

    print(f"Mean: {mean}, STD:{std}")

    input_data = (input_data - mean) / std
    target_data = (target_data - mean) / std

    prediction_window = 1000
    x = torch.from_numpy(input_data[-1].astype(np.float32)).cuda().reshape(1, -1)
    y = torch.from_numpy(target_data.astype(np.float32)).cuda().reshape(1, -1)
    y = model.predict(x, y, prediction_window)
    y = y.squeeze().cpu().numpy()

    # Denormalize the data
    input_data = (input_data * std) + mean
    y = (y * std) + mean
    original_data = np.append(input_data[:, 1].flatten(), input_data[-1, 1:].flatten())

    display_data(
        2,
        ["Original data", "Forecast data"],
        range(len(original_data)),
        original_data,
        range(len(original_data), len(original_data) + prediction_window),
        y[-prediction_window:],
    )

    torch.save(model.state_dict(), "models/autoregressive_model.pt")
    print("Model saved successfully")

    summary(
        model,
        input_size=(1, 10),
        col_names=("input_size", "output_size", "num_params"),
        verbose=0,
    )
