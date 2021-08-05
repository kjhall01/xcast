import sys
import datetime as dt

class ProgressBar:
	def __init__(self, total, step=1, label='PROGRESS: ', length=25):
		self.total = total
		self.label = label
		self.length = length
		self.step = step

	def show(self, count):
		if count % self.step == 0:
			stars = int( (count / self.total) * self.length)
			spaces = self.length - stars
			print(self.label + ' {} ['.format(dt.datetime.now()) +'*'*stars + ' '*spaces + ']', end = '\r')
			sys.stdout.flush()

	def finish(self):
		print(self.label + ' {} ['.format(dt.datetime.now()) +'*'*self.length + ']')
		sys.stdout.flush()
