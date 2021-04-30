# Predicting Stock Prices with Deep Learning

## Project Overview

**Deep learning** is part of a broader family of machine learning methods based on artificial neural networks, which are [inspired](https://en.wikipedia.org/wiki/Deep_learning) by our brain's own network of neurons. Among the popular deep learning paradigms, [**Long Short-Term Memory (LSTM)**](https://en.wikipedia.org/wiki/Long_short-term_memory) is a specialized archicture that can "memorize" patterns from historical sequences of data and extrapolate such patterns for future events. 

Since the stock market is naturally comprised of sequences of prices and volumes, more and more quantitative researchers and finance professionals are using LTSM to model and predict stock price movements. In this project, we will go through the end-to-end machine learning workflow of developing an LTSM model to predict stock market prices using PyTorch and Alpha Vantage APIs. 

The project is grouped into the following sections: 
- Data preparation: acquiring financial market data from Alpha Vantage
- Data preparation: noramlizing raw data
- Data preparation: generating training and validation datasets
- Defining the LSTM model
- Model training
- Model evaluation
- Predicting future stock prices

This tutorial has been written in a way such that all the essential code snippets have been embedded inline. You should be able to develop, train, and test your machine learning model without referring to other external pages or documents. 

Let's get started! 

## Data preparation: acquiring financial market data from Alpha Vantage

In this project, we will train an LSTM model to predict stock price movements. Before we can build the "crystal ball" to predict the future, however, we need historical stock price data to train our deep learning model. To this end, we will query the Alpha Vantage stock data API via a [popular python wrapper](https://github.com/RomelTorres/alpha_vantage). For the purpose of this project, we will obtain over 20 years of daily close prices for IBM from November 1999 to April 28, 2021. 

![historical prices](static/figure01-history-price.png)

<details>
<summary>View codes</summary>

```python
# Before proceeding, please make sure to: pip install alpha_vantage
from alpha_vantage.timeseries import TimeSeries 

def download_data(config):
    ts = TimeSeries(key=config["alpha_vantage"]["key"])
    data, meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])

    data_date = [date for date in data.keys()]
    data_date.reverse()

    data_close_price = [float(data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data.keys()] #note that we are using the ADJUSTED close field
    data_close_price.reverse()
    data_close_price = np.array(data_close_price)

    num_data_points = len(data_date)
    display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1]
    print("Number data points", num_data_points, display_date_range)

    return data_date, data_close_price, num_data_points, display_date_range

data_date, data_close_price, num_data_points, display_date_range = download_data(config)
```
</details>

Please note that we are using the **adjusted close** field of Alpha Vantage's [daily adjusted API](https://www.alphavantage.co/documentation/#dailyadj) to remove any artificial price turbulances due to stock splits and dividend payout events. It is generally considered an [industry best practice](http://www.crsp.org/products/documentation/crsp-calculations) to use split/dividend adjusted prices instead of raw prices to model stock price movements. 

## Data preparation: normalizing raw data

Machine learning algorithms that use [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) as an optimization technique require data to be scaled. This is due to the fact that the feature values in the model will affect the step size of the gradient descent, potentially skewing the LSTM model in unexpected ways. 

This is where **data normalization** comes in. Normalization can increase the accuracy of your model and help the gradient descent algorithm converge more quickly towards the target minima. By bringing the input data on the same scale and reducing its variance, none of the weights in the articial neural network will be wasted on normalizing tasks, which means the LSTM model can more efficiently learn from the data and store patterns in the network. Furthermore, LSTMs are intrinsically sensitive to the scale of the input data. For the above reasons, it is crucial to normalize the data.

Since stock prices can range from tens to hundreds and thousands - $40 to $160 in the case of IBM - we will perform normalization on the stock prices to narrow down the range of these values before feeding the data to the LSTM model. The following code snippets rescales the data so that it has a mean of 0 and the standard deviation is 1. 

<details>
<summary>View codes</summary>

```python
class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu

scaler = Normalizer()
normalized_data_close_price = scaler.fit_transform(data_close_price)
```
</details>

## Data preparation: generating training and validation datasets

[Supervised machine learning](https://en.wikipedia.org/wiki/Supervised_learning) methods such as LSTM learns the mapping function from input variables (x) to the output variable (Y). Learning from the training dataset can be thought of as a teacher supervising the learning process, as that teacher knows all the right answers. 

In this project, we will train the model to predict the 21<sup>st</sup> day price based on the past 20 days' close prices. The number of days, `20`, was selected based on a few reasons: 
- When LSTM models are used in natural language processing (NLP), the number of words in a sentence typically ranges from 15 to 20 words
- Gradient descent considerations: attempting to back-propagate across very long input sequences may result in vanishing gradients
- Longer sequences tend to have much longer training times

After transforming the dataset into input features and output labels, the shape of our `X` is `(5387, 20)`, 5387 for the number of rows, each row containing a sequence of past 20 days' prices. The corresponding `Y` data shape is ` (5387,)`, which matches the number of rows in `X`.

We also split the dataset into two parts, for training and validation. We split the data into 80:20 - 80% of the data is used for training, with the remaining 20% to verify our model's performance on predicting future prices. (Alternatively, another common practice is to split the initial data into train, validation, and test set (70/20/10), where the test dataset is not used at all during the training process.) After splitting our `X` and `Y` for training and validation, the size and dimension are as follows:

![dataset split](static/figure02-train-validation-split.png)

```python
data_x_train shape: (4309, 20)
data_y_train shape: (4309,)
data_x_val shape: (1078, 20)
data_y_val shape: (1078,)
```

<details>
<summary>View codes</summary>

```python
def prepare_data_x(x, window_size=20):
    # perform windowing
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row,window_size), strides=(x.strides[0],x.strides[0]))
    return output[:-1], output[-1]

data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=config["data"]["window_size"])

def prepare_data_y(x, window_size):
    # # perform simple moving average
    # output = np.convolve(x, np.ones(window_size), 'valid') / window_size
    output = x[window_size:]
    return output

data_y = prepare_data_y(normalized_data_close_price, window_size=config["data"]["window_size"])

# split dataset
split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
data_x_train = data_x[:split_index]
data_x_val = data_x[split_index:]
data_y_train = data_y[:split_index]
data_y_val = data_y[split_index:]

print("data_x_train shape", data_x_train.shape)
print("data_y_train shape", data_y_train.shape)
print("data_x_val shape", data_x_val.shape)
print("data_y_val shape", data_y_val.shape)
```
</details>

We will train our models using the [PyTorch](https://pytorch.org/), a machine learning library written in Python. At the heart of PyTorch's data loading utility is the [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html) class, an efficient data generation scheme that leverages the full potential of your computer's Graphics Processing Unit (GPU) during the training process. `DataLoader` requires the [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) object to define the loaded data. `Dataset` is a map-style dataset that implements the `__getitem__()` and `__len__()` protocols, and represents a map from indices to data samples. We will define the `Dataset` object and load it into the `DataLoader` as follows: 

<details>
<summary>View codes</summary>

```python
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)
```
</details>


## Defining the LSTM model

With the training and evaluation data now fully normalized and prepared, we are ready to build our first LSTM model! 

As mentioned before, LSTM is a specialized artificial neural network archicture that can "memorize" patterns from historical sequences of data and extrapolate such patterns for future events. Specifically, it belongs to a group of artifical neural networks called [Recurring Neural Networks (RNNs)](https://en.wikipedia.org/wiki/Recurrent_neural_network). 

LSTM is a popular artificial neural network because it manages to overcome many technical limitations of RNNs. For example, RNNs fail to learn when the data sequence is greater than 5 to 10 due to the [vanishing gradients problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem), where the gradients are vanishingly small, effectively preventing the model to learn. LSTMs can learn long sequences by enforcing constant error flow through self-connected hidden layers which contains memory cells and corresponding gate units. If you are interested in learning more about the inner workings of LSTM and RNNs, [this](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) is a great explainer for your reference. 

The artificial neural network defined has three main layers, with each layer designed with a specific logical purpose:
- linear layer 1 (`linear_1`): to map input values into a high dimensional feature space, transforming the features for the LSTM layer
- LSTM (`lstm`): to learn the data in sequence
- linear layer 2 (`linear_2`): to produce the predicted value based on LSTM's output

We added [Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html), which randomly zeroes some of the elements of the input data, therefore regularizing the network to prevent [overfitting](https://en.wikipedia.org/wiki/Overfitting) and improving overall model performance. As an optional step, we also initialize the LSTM's model weights, as some researchers have observed that it could help the model learn better. 

<details>
<summary>View codes</summary>

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(1, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # convert `x` into [batch, sequence, feature]
        x = torch.unsqueeze(x, 2) 

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)
        
        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1) 
        
        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:,-1]
```
</details>

## Model training

The LSTM model learns by iteratively making predictions given the training data `X`. We use [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) as the cost function, which measures the difference between the predicted values and the actual values. When the model is making bad predictions, the error rate will be high. The model will fine-tune its weights through [backpropagation](https://en.wikipedia.org/wiki/Backpropagation), improving its ability to make better predictions. Learning stops when the algorithm achieves an acceptable level of performance, where the cost function on the validation dataset is no longer showing incremental improvements. 

We used the [Adam optimizer](https://pytorch.org/docs/master/generated/torch.optim.Adam.html) [[paper](https://arxiv.org/abs/1412.6980)] that updates the model's parameters based on the learning rate through its `step()` method. This is how the model learns and fine-tunes its predictions. The learning rate controls how quickly the model converges. A learning rate that is too large can cause the model to converge too quickly to a suboptimal solution, whereas smaller learning rates require more training epochs given the smaller changes made to the weights each update. We also used the [StepLR scheduler](https://pytorch.org/docs/master/generated/torch.optim.lr_scheduler.StepLR.html) to reduce the learning rate during the training process. One could change to using [ReduceLROnPlateau](https://pytorch.org/docs/master/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) scheduler, which reduces the learning rate when a cost function has stopped improving for a "`patience`" number of epochs.

<details>
<summary>View codes</summary>

```python
def run_epoch(dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        bs = x.shape[0]

        x = x.to(config["training"]["device"])
        y = y.to(config["training"]["device"])

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / bs)

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr


train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)

model = LSTMModel()
model = model.to(config["training"]["device"])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

for epoch in range(config["training"]["num_epoch"]):
    loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
    loss_val, lr_val = run_epoch(val_dataloader)
    scheduler.step()
    
    print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
              .format(epoch+1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))
```

Console:

```
Epoch[1/100] | loss train:0.072375, test:0.002034 | lr:0.010000
Epoch[2/100] | loss train:0.012255, test:0.002809 | lr:0.010000
Epoch[3/100] | loss train:0.011213, test:0.001190 | lr:0.010000
Epoch[4/100] | loss train:0.010167, test:0.001058 | lr:0.010000
Epoch[5/100] | loss train:0.008862, test:0.001226 | lr:0.010000
Epoch[6/100] | loss train:0.008805, test:0.001040 | lr:0.010000
Epoch[7/100] | loss train:0.009633, test:0.001056 | lr:0.010000
Epoch[8/100] | loss train:0.008371, test:0.001223 | lr:0.010000
Epoch[9/100] | loss train:0.007957, test:0.001163 | lr:0.010000
Epoch[10/100] | loss train:0.008755, test:0.001218 | lr:0.010000
Epoch[11/100] | loss train:0.008608, test:0.001553 | lr:0.010000
Epoch[12/100] | loss train:0.007750, test:0.000989 | lr:0.010000
Epoch[13/100] | loss train:0.008037, test:0.001148 | lr:0.010000
Epoch[14/100] | loss train:0.008020, test:0.002444 | lr:0.010000
Epoch[15/100] | loss train:0.008136, test:0.001343 | lr:0.010000
Epoch[16/100] | loss train:0.007422, test:0.001240 | lr:0.010000
Epoch[17/100] | loss train:0.008435, test:0.001197 | lr:0.010000
Epoch[18/100] | loss train:0.007477, test:0.001124 | lr:0.010000
Epoch[19/100] | loss train:0.007947, test:0.001190 | lr:0.010000
Epoch[20/100] | loss train:0.008342, test:0.002806 | lr:0.010000
Epoch[21/100] | loss train:0.008021, test:0.001525 | lr:0.010000
Epoch[22/100] | loss train:0.008196, test:0.001073 | lr:0.010000
Epoch[23/100] | loss train:0.008553, test:0.001027 | lr:0.010000
Epoch[24/100] | loss train:0.008328, test:0.001063 | lr:0.010000
Epoch[25/100] | loss train:0.007597, test:0.001832 | lr:0.010000
Epoch[26/100] | loss train:0.008020, test:0.001408 | lr:0.010000
Epoch[27/100] | loss train:0.008042, test:0.001685 | lr:0.010000
Epoch[28/100] | loss train:0.007724, test:0.001103 | lr:0.010000
Epoch[29/100] | loss train:0.008303, test:0.001191 | lr:0.010000
Epoch[30/100] | loss train:0.008233, test:0.002329 | lr:0.010000
Epoch[31/100] | loss train:0.007013, test:0.001051 | lr:0.010000
Epoch[32/100] | loss train:0.007202, test:0.001056 | lr:0.010000
Epoch[33/100] | loss train:0.007504, test:0.001379 | lr:0.010000
Epoch[34/100] | loss train:0.007983, test:0.001071 | lr:0.010000
Epoch[35/100] | loss train:0.007540, test:0.001208 | lr:0.010000
Epoch[36/100] | loss train:0.006941, test:0.001141 | lr:0.010000
Epoch[37/100] | loss train:0.008131, test:0.001175 | lr:0.010000
Epoch[38/100] | loss train:0.007338, test:0.001169 | lr:0.010000
Epoch[39/100] | loss train:0.007214, test:0.001075 | lr:0.010000
Epoch[40/100] | loss train:0.007625, test:0.001993 | lr:0.010000
Epoch[41/100] | loss train:0.006754, test:0.001013 | lr:0.001000
Epoch[42/100] | loss train:0.006540, test:0.001075 | lr:0.001000
Epoch[43/100] | loss train:0.005956, test:0.001227 | lr:0.001000
Epoch[44/100] | loss train:0.006048, test:0.000997 | lr:0.001000
Epoch[45/100] | loss train:0.006459, test:0.001006 | lr:0.001000
Epoch[46/100] | loss train:0.006255, test:0.001063 | lr:0.001000
Epoch[47/100] | loss train:0.006164, test:0.001027 | lr:0.001000
Epoch[48/100] | loss train:0.006233, test:0.001121 | lr:0.001000
Epoch[49/100] | loss train:0.006505, test:0.001039 | lr:0.001000
Epoch[50/100] | loss train:0.006216, test:0.000984 | lr:0.001000
Epoch[51/100] | loss train:0.006158, test:0.001035 | lr:0.001000
Epoch[52/100] | loss train:0.006408, test:0.001026 | lr:0.001000
Epoch[53/100] | loss train:0.005853, test:0.001000 | lr:0.001000
Epoch[54/100] | loss train:0.006474, test:0.001047 | lr:0.001000
Epoch[55/100] | loss train:0.006450, test:0.000979 | lr:0.001000
Epoch[56/100] | loss train:0.006138, test:0.001252 | lr:0.001000
Epoch[57/100] | loss train:0.006569, test:0.001084 | lr:0.001000
Epoch[58/100] | loss train:0.006730, test:0.001000 | lr:0.001000
Epoch[59/100] | loss train:0.006063, test:0.001017 | lr:0.001000
Epoch[60/100] | loss train:0.006475, test:0.001034 | lr:0.001000
Epoch[61/100] | loss train:0.006290, test:0.001090 | lr:0.001000
Epoch[62/100] | loss train:0.006195, test:0.001049 | lr:0.001000
Epoch[63/100] | loss train:0.006205, test:0.000986 | lr:0.001000
Epoch[64/100] | loss train:0.005922, test:0.001058 | lr:0.001000
Epoch[65/100] | loss train:0.006228, test:0.000987 | lr:0.001000
Epoch[66/100] | loss train:0.006276, test:0.001069 | lr:0.001000
Epoch[67/100] | loss train:0.006421, test:0.001023 | lr:0.001000
Epoch[68/100] | loss train:0.006610, test:0.001169 | lr:0.001000
Epoch[69/100] | loss train:0.006277, test:0.001188 | lr:0.001000
Epoch[70/100] | loss train:0.006136, test:0.001002 | lr:0.001000
Epoch[71/100] | loss train:0.006062, test:0.001000 | lr:0.001000
Epoch[72/100] | loss train:0.006437, test:0.001015 | lr:0.001000
Epoch[73/100] | loss train:0.006237, test:0.001029 | lr:0.001000
Epoch[74/100] | loss train:0.006481, test:0.001044 | lr:0.001000
Epoch[75/100] | loss train:0.006396, test:0.001066 | lr:0.001000
Epoch[76/100] | loss train:0.006264, test:0.001004 | lr:0.001000
Epoch[77/100] | loss train:0.006077, test:0.001053 | lr:0.001000
Epoch[78/100] | loss train:0.006301, test:0.001015 | lr:0.001000
Epoch[79/100] | loss train:0.006138, test:0.001036 | lr:0.001000
Epoch[80/100] | loss train:0.006425, test:0.001002 | lr:0.001000
Epoch[81/100] | loss train:0.006095, test:0.000978 | lr:0.000100
Epoch[82/100] | loss train:0.006089, test:0.000995 | lr:0.000100
Epoch[83/100] | loss train:0.006163, test:0.001047 | lr:0.000100
Epoch[84/100] | loss train:0.006165, test:0.000989 | lr:0.000100
Epoch[85/100] | loss train:0.006059, test:0.001013 | lr:0.000100
Epoch[86/100] | loss train:0.005984, test:0.000994 | lr:0.000100
Epoch[87/100] | loss train:0.006035, test:0.001007 | lr:0.000100
Epoch[88/100] | loss train:0.005993, test:0.000997 | lr:0.000100
Epoch[89/100] | loss train:0.005973, test:0.000981 | lr:0.000100
Epoch[90/100] | loss train:0.006219, test:0.001013 | lr:0.000100
Epoch[91/100] | loss train:0.006433, test:0.000993 | lr:0.000100
Epoch[92/100] | loss train:0.006332, test:0.000976 | lr:0.000100
Epoch[93/100] | loss train:0.006214, test:0.000996 | lr:0.000100
Epoch[94/100] | loss train:0.006101, test:0.000981 | lr:0.000100
Epoch[95/100] | loss train:0.006204, test:0.000973 | lr:0.000100
Epoch[96/100] | loss train:0.005765, test:0.000990 | lr:0.000100
Epoch[97/100] | loss train:0.006165, test:0.000986 | lr:0.000100
Epoch[98/100] | loss train:0.006106, test:0.000985 | lr:0.000100
Epoch[99/100] | loss train:0.006233, test:0.001011 | lr:0.000100
Epoch[100/100] | loss train:0.005894, test:0.000996 | lr:0.000100
```
</details>

## Model evaluation

To visually inspect our model's performance, we use the trained model to make predictions with the training and validation dataset. If we see that the model can predict values that closely mirror the training dataset, it shows that the model managed to memorize the data. And if the model can predict values that resemble the validation dataset, it has managed to learn the pattern of our sequential data and predict unseen data points. If you have split the data into train, validation, and test set, plotting and comparing against the test set will give you a good indicator on the model's performance.

![actual vs predicted](static/figure03-actual-vs-predicted.png)

From our results, the model manages to learn and predict on both training and validation datasets very well, as the `Predicted Train` and `Predicted Validation` lines significantly overlap with the `Actual` values. 

Let's zoom into the chart and look closely at `Predicted Validation`, comparing against `Actual` values.

![predicted validation zoom in](static/figure04-actual-vs-predicted-zoom.png)

What a beautiful graph! 

It is also worth noting that model training & evaluation is an iterative process. Please feel free to go back to the "model training" step to fine-tune the model and re-evaluate the model to see if there is further performance boost. 

## Predicting future stock prices

By now, we have trained an LSTM model that can (fairly accurately) predict the next day's price based on the past 20 days' close prices. This means we now have a crystall ball in hand! At the time of writing this tutorial, tomorrow is April 29, 2021. Let's supply the past 20 days' close prices to the model and...

![predicted tomorrow price](static/figure05-predict-the-unseen.png)

The model predicts that IBM's close price on April 29, 2021 is $143.01 per share. Is the prediction good enough? How about other stocks such as TSLA, APPL, or the hugely popular Gamestop stock GME? Beyond the close prices, are there any other external data we can feed to the LSTM model to make it even more robust? We will now pass the baton to you, our fearless reader! 

<details>
<summary>View codes</summary>

```python
# predict on the unseen data, tomorrow's price 

model.eval()

x = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0) # this is the data type and shape required
prediction = model(x)
prediction = prediction.detach().numpy()

print("Tomorrow's price:", round(to_plot_data_y_test_pred[plot_range-1], 2))
```
</details>

**Disclaimer: this content is for educational purposes only and does NOT constitute investment advice. **
