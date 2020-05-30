import random

with open("hyperparameters.txt", "w") as f:
	print("Epoch={}".format(random.randint(1, 20)), file=f)
	print("CRP_Layers={}".format(random.randint(1,9)), file=f)
	print("Hidden_Layers={}".format(random.randint(1,9)), file=f)
	print("kernel_size={}".format(random.choice(("(2,2)","(3,3)","(4,4)","(5,5)","(6,6)"))), file=f)
	print("Filters={}".format(random.randrange(32, 1024, 16)), file=f)
	print("Neurons={}".format(random.randrange(2, 1024, 2)), file=f)
	print("Optimizers={}".format(random.choice(("Adam()","RMSprop()","SGD()","Nadam()","Adamax()","Adadelta()"))), file=f)
	print("Pooling={}".format(random.choice(("MaxPooling2D()", "AveragePooling2D()"))), file=f)

