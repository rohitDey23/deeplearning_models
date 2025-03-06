import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

np.random.seed(2)


class SimpleRNN(nn.Module):
    def __init__(
        self,
        rnn_layers,
        num_features,
        hidden_nodes,
        out_features,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = self.model_construction(
            rnn_layers, num_features, hidden_nodes, out_features
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()

        self.training_losses = []
        self.testing_losses = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def forward(self, input_data):
        output, hidden = self.model[0](input_data)
        output = self.model[1](output[:, -1])
        return output

    def model_construction(self, rnn_layers, num_features, hidden_size, out_features):
        rnn_model = nn.RNN(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            nonlinearity="relu",
            batch_first=True,
        )

        model = nn.ModuleList([rnn_model, nn.Linear(hidden_size, out_features)])

        return model

    def split_data(self, input_data, target_data, sr=0.5):
        split = int(0.5 * len(input_data))
        input_traindata = input_data[:split]
        target_traindata = target_data[:split]
        input_traindata = torch.from_numpy(input_traindata.astype(np.float32))
        target_traindata = torch.from_numpy(target_traindata.astype(np.float32))

        input_testdata = input_data[split:]
        target_testdata = target_data[split:]
        input_testdata = torch.from_numpy(input_testdata.astype(np.float32))
        target_testdata = torch.from_numpy(target_testdata.astype(np.float32))

        return input_traindata, target_traindata, input_testdata, target_testdata

    def train_model(self, input_data, target_data, num_epochs=400):
        training_losses = []
        testing_losses = []

        # Split the data into training and testing
        input_traindata, target_traindata, input_testdata, target_testdata = (
            self.split_data(input_data, target_data, 0.75)
        )
        # Trandfer the data to the device
        input_traindata = input_traindata.to(self.device)
        target_traindata = target_traindata.to(self.device)
        input_testdata = input_testdata.to(self.device)
        target_testdata = target_testdata.to(self.device)

        # Training loop
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            output = self.forward(input_traindata)
            loss = self.loss_fn(output, target_traindata)
            # Backward pass
            loss.backward()
            self.optimizer.step()
            # Append the loss to the list
            training_losses.append(loss.item())

            ## Testing the model
            with torch.no_grad():
                test_output = self.forward(input_testdata)
                test_loss = self.loss_fn(test_output, target_testdata)
                testing_losses.append(test_loss.item())

            if epoch % 100 == 0:
                print(
                    f"Epoch: {epoch}, Train Loss: {training_losses[-1]}, Test Loss: {testing_losses[-1]}"
                )

        return training_losses, testing_losses

    def predict(self, input_data, reference_window, prediction_window):
        # Input data should a simple time series data of length > reference_window
        predicted_data = []
        # Reshape the input data for the model
        input_data = input_data[-reference_window:].reshape(-1, reference_window, 1)
        input_data = torch.from_numpy(input_data.astype(np.float32)).to(self.device)
        # Set the model to evaluation
        with torch.no_grad():
            for i in range(prediction_window):
                output = self.forward(input_data)
                predicted_data.append(output.item())

                # Update the input data such that number of elements = reference_window
                output = torch.reshape(output, (1, 1, 1))
                input_data = torch.cat((input_data, output), dim=1)
                input_data = input_data[:, -reference_window:]
                input_data = torch.reshape(input_data, (1, reference_window, 1))

        return np.array(predicted_data)


class CustomDataGenerator:
    def __init__(self, amplitude, noise, num_points, t, ts_length):
        self.amplitude = amplitude
        self.noise = noise
        self.num_points = num_points
        self.timeWindow = t
        self.ts_length = ts_length

    def generate_data(self, if_show=False):
        time_series = np.linspace(0, self.timeWindow, self.num_points)
        self.data = self.amplitude * np.sin(time_series) + np.random.normal(
            0, self.noise, size=self.num_points
        )  # SINE wave A=10 and with noise (0, 0.2)

        if if_show:
            self.display_data(
                1,
                [f"Simulated Sine Wave ({self.amplitude}, 1,({0, self.noise}))"],
                time_series,
                self.data,
            )

        return self.data

    def structure_data(self, ts_length):
        input_data = []
        target_data = []
        for sample in range(len(self.data) - ts_length):
            input_data.append(self.data[sample : sample + ts_length])
            target_data.append(self.data[sample + ts_length])

        input_data = np.array(input_data).reshape(-1, ts_length, 1)
        target_data = np.array(target_data).reshape(-1, 1)

        return input_data, target_data

    @staticmethod
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


if __name__ == "__main__":
    ## Generate the data
    data_gen = CustomDataGenerator(20, 0.2, 1100, 100, 10)
    data = data_gen.generate_data(True)
    print(f"Data shape generated: {data.shape}")

    # Reshape the data
    ts_length = 10
    input_data, target_data = data_gen.structure_data(ts_length)
    print(
        f"Data restructured to: Input: {input_data.shape} | Target: {target_data.shape}"
    )

    # Initialize the RNN model
    rnn = SimpleRNN(rnn_layers=1, num_features=1, hidden_nodes=15, out_features=1)

    # Train the model
    train_loss, test_loss = rnn.train_model(input_data, target_data, 5000)

    # Display the loss display functions
    data_gen.display_data(
        2,
        ["training loss", "testing loss"],
        range(len(train_loss)),
        train_loss,
        range(len(test_loss)),
        test_loss,
    )

    predicted_data = rnn.predict(data, ts_length, prediction_window=1000)
    print(f"Predicted data shape: {predicted_data.shape}")

    # Display the predicted data
    data_gen.display_data(
        2,
        ["Original Data", "Predicted Data"],
        range(len(data)),
        data,
        range(len(data), len(predicted_data) + len(data)),
        predicted_data,
    )

    torch.save(rnn.state_dict(), "models/rnn_model.pt")
    print("Model saved successfully")

    summary(
        rnn,
        input_size=(1, 10, 1),
        col_names=("input_size", "output_size", "num_params"),
        verbose=1,
    )
