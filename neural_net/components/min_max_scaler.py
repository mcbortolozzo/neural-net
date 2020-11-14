
class MinMaxScaler():

	def fit(self, data):
		self.data_max = data.max()
		self.data_min = data.min()

	def transform(self, data):
		return (data-self.data_min)/(self.data_max-self.data_min)

	def reverse(self, data):
		return data*(self.data_max - self.data_min) + self.data_min