import dill


def load_option(file_name):
    input_file = open("src/options/" + file_name + ".pkl", 'rb')
    return dill.load(input_file)


def save_option(file_name, option):
    print(file_name)
    output = open("src/options/" + file_name + ".pkl", 'wb')
    dill.dump(option, output)
    output.close()