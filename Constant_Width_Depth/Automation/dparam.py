"""
abstract container class
cam be used to encapsulate any data
"""

class Dparam:
    def __init__(self):
        pass

    def save(self, fname="param_save.dat"):
        """
        save all attribute values to a file
        :param fname:
        :return:
        """
        with open(fname,"w") as fileout:
            for key in self.__dict__:
                fileout.write("%s = %s\n" % (key, self.__dict__[key]))

if __name__ == "__main__":

    def add_var(par):
        par.batch_size = 200
        par.early_stop_patience = 500
        par.max_epochs = 20000

    my_param = Dparam()
    my_param.layers = [ 2, 10, 10, 1 ]
    my_param.activations = [ 'None', 'tanh', 'tanh', 'linear' ]
    my_param.Ck = 1

    add_var(my_param)

    param_file = './data/test_param.dat'
    my_param.save(param_file)



