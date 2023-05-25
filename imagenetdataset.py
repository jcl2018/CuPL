
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset
import os
from imagenet_classnames_cn.sense_to_idx import sense
from imagenet_classnames_cn.imagenet_classes_cn import imagenet_classes_cn


class ImagenetDataset(Dataset):
	def __init__(self, path, transform=None):
		self.transform = transform
		self.paths = []
		self.labels = []
		self.label_to_idx = defaultdict(int)
		self.idx_to_label = []
		self.idx_to_text = []


		sense_to_name = {}
		i = 0
		for item in sense:
			sense_num = "n" + sense[item]['id'].split("-")[0]
			sense_to_name[sense_num] = imagenet_classes_cn[i]
			i += 1

		for directory in os.listdir(path):
			First = True
			f = os.path.join(path, directory)
			if os.path.isdir(f):
				for image in os.listdir(f):
					ext = image.split('.')[-1]
					if ext == 'JPEG':
						image_path = os.path.join(f, image)
						self.paths.append(image_path)
						label_name = sense_to_name[directory]
						if First:
							First = False
							self.label_to_idx[directory] = len(self.idx_to_label)
							self.idx_to_label.append(directory)
							self.idx_to_text.append(label_name)
						self.labels.append(self.label_to_idx[directory])

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, i):
		img, label = Image.open(self.paths[i]), int(self.labels[i])
		if self.transform is not None:
			img = self.transform(img)
		return img, label, i


class ImagenetDatasetCN(Dataset):
	def __init__(self, path, transform=None):
		self.transform = transform
		self.paths = []
		self.labels = []
		self.label_to_idx = defaultdict(int)
		self.idx_to_label = []
		self.idx_to_text = []

		sense_to_name = {}
		i = 0
		for item in sense:
			sense_num = "n" + sense[item]['id'].split("-")[0]
			sense_to_name[sense_num] = imagenet_classes_cn[i]
			i += 1

		for directory in os.listdir(path):
			First = True
			f = os.path.join(path, directory)
			if os.path.isdir(f):
				for image in os.listdir(f):
					ext = image.split('.')[-1]
					if ext == 'JPEG':
						image_path = os.path.join(f, image)
						self.paths.append(image_path)
						label_name = sense_to_name[directory]  # EN label name list
						if First:
							First = False
							self.label_to_idx[directory] = len(self.idx_to_label)
							self.idx_to_label.append(directory)  # [n111111, n22222]
							self.idx_to_text.append(label_name)  # [big white shartk, gold fish, ..] -> CN todo
						self.labels.append(self.label_to_idx[directory])  # [0, 1, 2]

		for idx in range(1000):
			self.idx_to_text.append(imagenet_classes_cn[idx])

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, i):
		img, label = Image.open(self.paths[i]), int(self.labels[i])
		if self.transform is not None:
			img = self.transform(img)
		return img, label, i
