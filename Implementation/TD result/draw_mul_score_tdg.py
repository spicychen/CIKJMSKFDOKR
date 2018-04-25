import pandas as pd
import matplotlib.pyplot as plt

cont1 = 'lr_'
cont2 = '_df_'
cont3 = '_etdf_'
cont4 = '_nhid_'
file_name = 'tdG_change_params'

lrs = [1,
0.8,
0.5,
0.2,
0.1,
0.05,
0.01,
0.001,
0.0001,
0,
]

dfs = [1,
0.95,
0.9,
0.8,
0.6,
0.4,
0.2,
0.1,
0.01,
0,
]
	
ets = [1,
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

num_of_hid=[20,
18,
16,
14,
12,
10,
8,
6,
4,
2,
]
	
for i in range(1,11):
	#print(i)
	lr = lrs[i-1]
	df = 0.95
	lam = 0.7
	rname = cont1 + ('%.4f' % lr) + cont2 + ('%.4f' % df) + cont3 + ('%.4f' % lam) + '.gz'
	fig = pd.read_pickle(rname, compression = 'gzip')

	fig = fig.groupby(fig.index//20).mean()
#fig = fig.interpolate(method = 'cubic')
	ax = plt.subplot(10,4, i*4-3)
	#plt.title('lr: '+ ('%.4f' % lr))
	plt.text(0.5,0.5,(r'$\alpha=%.4f$' % lr),fontdict=dict(color='r'),horizontalalignment='center',
		verticalalignment='center',transform=ax.transAxes)
	plt.plot(fig)
	
for i in range(2,6):
	lr = 0.1
	df = dfs[i-1]
	lam = 0.7
	nhid = 5
	rname = cont1 + ('%.4f' % lr) + cont2 + ('%.4f' % df) + cont3 + ('%.4f' % lam) + cont4 + ('%d' % nhid) + '.gz'
	fig = pd.read_pickle(rname, compression = 'gzip')

	fig = fig.groupby(fig.index//20).mean()

	ax = plt.subplot(10,4, i*4-2)
	#plt.title('df: '+ ('%.4f' % df))
	plt.text(0.5,0.5,(r'$\lambda=%.4f$' % df),fontdict=dict(color='r'),horizontalalignment='center',
		verticalalignment='center',transform=ax.transAxes)
	plt.plot(fig)
	
for i in range(1,11):
	lr = 0.1
	df = 0.95
	lam = ets[i-1]
	nhid = 5
	rname = cont1 + ('%.4f' % lr) + cont2 + ('%.4f' % df) + cont3 + ('%.4f' % lam) + cont4 + ('%d' % nhid) + '.gz'
	fig = pd.read_pickle(rname, compression = 'gzip')

	fig = fig.groupby(fig.index//20).mean()

	ax = plt.subplot(10,4, i*4-1)
	#plt.title('eps: '+ ('%.4f' % eps))
	plt.text(0.5,0.5,(r'$ET\lambda=%.4f$' % lam),fontdict=dict(color='r'),horizontalalignment='center',
		verticalalignment='center',transform=ax.transAxes)
	plt.plot(fig)

for i in range(1,11):
	lr = 0.1
	df = 0.95
	lam = 0.7
	nhid = num_of_hid[i-1]
	rname = cont1 + ('%.4f' % lr) + cont2 + ('%.4f' % df) + cont3 + ('%.4f' % lam) + cont4 + ('%d' % nhid) + '.gz'
	fig = pd.read_pickle(rname, compression = 'gzip')

	fig = fig.groupby(fig.index//20).mean()

	ax = plt.subplot(10,4, i*4)
	#plt.title('eps: '+ ('%.4f' % eps))
	plt.text(0.5,0.5,(r'$nhid=%.4f$' % nhid),fontdict=dict(color='r'),horizontalalignment='center',
		verticalalignment='center',transform=ax.transAxes)
	plt.plot(fig)

plt.subplots_adjust(left  = 0.125,right = 0.9,bottom = 0.1,top = 0.9,wspace = 0.2,hspace = 0.5)
#plt.suptitle(file_name)
win = plt.gcf()
win.canvas.set_window_title(file_name)
plt.show()