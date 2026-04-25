import matplotlib.pyplot as plt


def plot_loss(values, title="Loss"):
    plt.figure(figsize=(6, 4))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()