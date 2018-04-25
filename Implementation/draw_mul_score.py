import pandas as pd
import matplotlib.pyplot as plt

cont1 = 'lr_'
cont2 = '_df_'
cont3 = '_eps_'
file_name = 'lr_change'

lrs = [0,
0.0001,
0.001,
0.01,
0.05,
0.1,
0.2,
0.5,
0.8,
1,
]

dfs = [1,
0.9,
0.8,
0.6,
0.4,
0.2,
0.1,
0.01,
0.001,
0,
]
	
e_gs = [1,
0.9,
0.8,
0.7,
0.6,
0.5,
0.4,
0.3,
0.2,
0.1,
]

for i in range(1,11):
	#print(i)
	lr = lrs[i-1]
	df = 0.9
	eps = 0.9
	rname = cont1 + ('%.4f' % lr) + cont2 + ('%.4f' % df) + cont3 + ('%.4f' % eps) + '.gz'
	fig = pd.read_pickle(rname, compression = 'gzip')

	fig = fig.groupby(fig.index//20).mean()
#fig = fig.interpolate(method = 'cubic')
	ax = plt.subplot(10,3, i*3-2)
	#plt.title('lr: '+ ('%.4f' % lr))
	plt.text(0.5,0.5,(r'$\alpha=%.4f$' % lr),fontdict=dict(color='r'),horizontalalignment='center',
		verticalalignment='center',transform=ax.transAxes)
	plt.plot(fig)
	
for i in range(1,11):
	lr = 0.01
	df = dfs[i-1]
	eps = 0.9
	rname = cont1 + ('%.4f' % lr) + cont2 + ('%.4f' % df) + cont3 + ('%.4f' % eps) + '.gz'
	fig = pd.read_pickle(rname, compression = 'gzip')

	fig = fig.groupby(fig.index//20).mean()

	ax = plt.subplot(10,3, i*3-1)
	#plt.title('df: '+ ('%.4f' % df))
	plt.text(0.5,0.5,(r'$\lambda=%.4f$' % df),fontdict=dict(color='r'),horizontalalignment='center',
		verticalalignment='center',transform=ax.transAxes)
	plt.plot(fig)
	
for i in range(1,11):
	lr = 0.01
	df = 0.9
	eps = e_gs[i-1]
	rname = cont1 + ('%.4f' % lr) + cont2 + ('%.4f' % df) + cont3 + ('%.4f' % eps) + '.gz'
	fig = pd.read_pickle(rname, compression = 'gzip')

	fig = fig.groupby(fig.index//20).mean()

	ax = plt.subplot(10,3, i*3)
	#plt.title('eps: '+ ('%.4f' % eps))
	plt.text(0.5,0.5,(r'$\epsilon=%.4f$' % eps),fontdict=dict(color='r'),horizontalalignment='center',
		verticalalignment='center',transform=ax.transAxes)
	plt.plot(fig)

plt.subplots_adjust(left  = 0.125,right = 0.9,bottom = 0.1,top = 0.9,wspace = 0.2,hspace = 0.5)
#plt.suptitle(file_name)
win = plt.gcf()
win.canvas.set_window_title(file_name)
plt.show()