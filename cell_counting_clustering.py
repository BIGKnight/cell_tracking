import torch.nn as nn
import cv2
import torch
import numpy as np


class Binarization(nn.Module):
    def __init__(self, dir, height, width, batch):
        super(Binarization, self).__init__()
        maps = []
        for i in range(batch):
            img = cv2.imread(dir + str(i) + ".png")
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = torch.FloatTensor(img_gray)
            maps.append(img_gray)
        self.batch = len(maps)
        self.height = height
        self.width = width
        self.binarization_maps = torch.stack(maps)

    def forward(self, alpha):
        threshold = 255 * alpha
        indicator_maps = 1 - self.binarization_maps.div(threshold).floor().clamp(0, 1)
        nuc = [[] for i in range(self.batch)]
        for n in range(self.batch):
            for i in range(self.height):
                for j in range(self.width):
                    if indicator_maps[n][i][j] > 0:
                        nuc[n].append((i, j))
        return indicator_maps, nuc


class Hierachy_Clustor(object):
    def __init__(self, binarization_maps, Mnuc, height, width):
        self.binarization_maps = binarization_maps
        self.Mnuc = Mnuc
        self.is_visited = None
        self.batch = len(binarization_maps)
        self.height = height
        self.width = width
        self.clusters = None

    def hierachy_clustering(self):
        self.clusters = [[] for i in range(self.batch)]
        for i in range(self.batch):
            current_bi_map = self.binarization_maps[i]
            current_Mnuc = self.Mnuc[i]
            self.is_visited = np.zeros([240, 320])
            cluster_index = 0

            for x, y in current_Mnuc:
                if self.is_visited[x][y] == 0:
                    self.clusters[i].append([])
                    self.broad_first_searching(current_bi_map, x, y, i, cluster_index)
                    cluster_index += 1
        return self.clusters

    def broad_first_searching(self, binary_map, i, j, batch_index, cluster_index):
        binary_value = binary_map[i][j]
        if binary_value == 0 or self.is_visited[i][j] == 1:
            return
        else:
            self.is_visited[i][j] = 1
            self.clusters[batch_index][cluster_index].append([i, j])

            if i + 1 <= self.height - 1 and self.is_visited[i + 1][j] == 0:
                self.broad_first_searching(binary_map, i + 1, j, batch_index, cluster_index)

            if j + 1 <= self.width - 1 and self.is_visited[i][j + 1] == 0:
                self.broad_first_searching(binary_map, i, j + 1, batch_index, cluster_index)

            if i - 1 >= 0 and self.is_visited[i - 1][j] == 0:
                self.broad_first_searching(binary_map, i - 1, j, batch_index, cluster_index)

            if j - 1 >= 0 and self.is_visited[i][j - 1] == 0:
                self.broad_first_searching(binary_map, i, j - 1, batch_index, cluster_index)
            return
