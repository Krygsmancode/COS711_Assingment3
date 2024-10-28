import matplotlib.pyplot as plt
import numpy as np

def evaluate_overfitting(history, threshold: float = 0.1, epochs_threshold = 10, starting_epoch: int = 0):
    """
    Evaluates if the model is overfitting or underfitting based on the training and validation loss.
    
    Parameters:
    - history: The history object returned by model.fit(), which contains the loss values.
    - threshold: The acceptable difference between training and validation loss to determine overfitting (default 0.1).
    
    Returns:
    - returns whether the model is overfitting, underfitting, or neither.
    """
    # Extract training and validation loss
    train_loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    running_diff = []
    end_piont = starting_epoch + epochs_threshold
    starting_point = starting_epoch
    if starting_epoch < 0:
        starting_point = 0
    if starting_epoch >= len(train_loss):
        starting_point = len(train_loss)-1
    if end_piont >= len(train_loss):
        end_piont = len(train_loss)-1
    for i in range(starting_point, end_piont):
        running_diff.append(abs(train_loss[i] - val_loss[i]))

    if len(running_diff) == 0:
        return False
    return np.mean(running_diff) > threshold


def find_overfitting_piont(history, threshold: float = 0.1, epochs_threshold = 10):
    """
    Evaluates if the model is overfitting or underfitting based on the training and validation loss
    and finds the piont at which the model starts to overfit.
    
    Parameters:
    - history: The history object returned by model.fit(), which contains the loss values.
    - threshold: The acceptable difference between training and validation loss to determine overfitting (default 0.1).
    
    Returns:
    - A integer representing the epoch at which the model starts to overfit. (-1 if the model is not overfitting)
    """
    # Extract training and validation loss
    train_loss = history['loss']
    val_loss = history['val_loss']

    if epochs_threshold > len(train_loss):
        print(f"epochs_threshold is greater than the number of epochs in the history object.")
        return -1

    overfitting_epoch = -1
    started_overfitting = False
    stopped_overfitting = False
    stopped_overfitting_epoch = -1
    for i in range(0, len(train_loss)-1):
        if not started_overfitting and stopped_overfitting:
            if i < stopped_overfitting_epoch:
                continue
            else:
                stopped_overfitting = False
                stopped_overfitting_epoch = -1

        # print (f"Epoch {i}")
        try:
            if evaluate_overfitting(history, threshold=threshold, epochs_threshold=epochs_threshold, starting_epoch=i):
                started_overfitting = True
                overfitting_epoch = i
                last_overfitting_state = True
                last_overfitting_epoch = i
                for j in range(i, len(train_loss)-epochs_threshold-1):
                    new_epochs_threshold = epochs_threshold + j
                    if new_epochs_threshold+i >= len(train_loss):
                        new_epochs_threshold = len(train_loss)-i-1
                        
                    if evaluate_overfitting(history, threshold=threshold, epochs_threshold=new_epochs_threshold, starting_epoch=i):
                        last_overfitting_state = True
                        last_overfitting_epoch = j+i
                    else:
                        last_overfitting_state = False

                    if not last_overfitting_state:
                        if last_overfitting_epoch-j+i > epochs_threshold:
                            # stopped overfitting
                            overfitting_epoch = -1
                            stopped_overfitting = True
                            stopped_overfitting_epoch = j+i
                            break
        except:
            print(f"Error evaluating overfitting at epoch {i}.")
            continue

    if overfitting_epoch == -1:
        # print("Model is not overfitting.")
        return -1
    else:
        # print(f"Model is overfitting at epoch {overfitting_epoch}.")
        return overfitting_epoch
    
def plot_history_and_overfitting(history, threshold: float = 0.1, epochs_threshold = 10):
    """
    Plots the training and validation loss over epochs and highlights the point where the model starts to overfit.
    
    Parameters:
    - history: The history object returned by model.fit(), which contains the loss values.
    - threshold: The acceptable difference between training and validation loss to determine overfitting (default 0.1).
    """
    print("Plotting history and overfitting...")

    # Extract training and validation loss
    train_loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    
    # Find the point where the model starts to overfit
    overfitting_epoch = find_overfitting_piont(history, threshold, epochs_threshold)
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 4))
    plt.plot(epochs, train_loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    print(f"Overfitting epoch: {overfitting_epoch}")
    
    # Highlight the point where the model starts to overfit
    if overfitting_epoch != -1:
        plt.axvline(x=overfitting_epoch, color='g', linestyle='--', label='Overfitting')
    
    plt.show()