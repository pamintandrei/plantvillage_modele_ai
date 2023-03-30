def write_results(filename, epoch, loss, accuracy, top5):
    with open(filename, "a") as file:
        file.write(f"{epoch}, {loss}, {accuracy}, {top5} \n")