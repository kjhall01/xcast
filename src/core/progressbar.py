import sys
import datetime as dt

class ProgressBar:
	"""Progress Bar Class
	---------------------------------------------
	Prints 'LABEL: [****       ]' each time .show(n) is called
	where the number of *'s is equal to n / total * length
	_____________________________________________
	"""

	def __init__(self, total, step=1, label='PROGRESS: ', length=25):
		self.total = total
		self.label = label
		self.length = length
		self.step = step
		self.start = dt.datetime.now()

	def show(self, count):
		"""prints progressbar at count / total * length stars progress"""
		if count % self.step == 0:
			stars = int( (count / self.total) * self.length)
			spaces = self.length - stars
			print(self.label + ' [' +'*'*stars + ' '*spaces + '] ({}/{}) {}'.format(count, self.total, dt.datetime.now()-self.start), end = '\r')
			sys.stdout.flush()

	def finish(self):
		"""prints full progressbar and progresses to next line"""
		print(self.label + ' [' +'*'*self.length + '] ({}/{}) {}'.format(self.total, self.total, dt.datetime.now()-self.start))
		sys.stdout.flush()
