import matplotlib.pyplot

matplotlib.pyplot.plotfile('err.csv', ('iteration','err','validation_err'),subplots=False, 
                              plotfuncs={'err': 'loglog','validation_err':'loglog'},
							  linestyle=':',linewidth=4)
matplotlib.pyplot.grid(which="both")
matplotlib.pyplot.title('Error vs iteration')
matplotlib.pyplot.savefig('err.png')
matplotlib.pyplot.show()

