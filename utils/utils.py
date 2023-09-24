import numpy as np
import torch
import torch.nn as nn

def get_data_settings(data):
  if data == 'uci':
    partition_size = 200
    last = 0
  elif data == 'btc_otc':
    partition_size = 100
    last = 0
  elif data == 'email_dnc':
    partition_size = 100
    last = 0
  return partition_size, last

class GenFGANLoss(nn.Module):
    def __init__(self, alpha_= 0.1, beta_=15, **kwargs):
        super().__init__()
        self.alpha = alpha_
        self.beta = beta_

    def forward(self, d_out, g_out):
        # calculate loss using the function defined in the paper
        bce = torch.nn.BCELoss()
        EL = bce(d_out, self.alpha * torch.ones_like(d_out))
        mu =  g_out - torch.ones_like(g_out) * g_out.mean(dim=0) 
        
        DL = 0
        for i in range(3):
          # DL += 2*torch.norm(mu[:,i], p=2) / (max(g_out[:,i])-min(g_out[:,i]))
          DL += torch.norm(mu[:,i], p=2) / torch.mean(g_out[:,i])
        DL = DL/3
        loss = EL + self.beta * 1/DL
        return loss

class DiscFGANLoss(nn.Module):
    def __init__(self, gamma_=1, **kwargs):
        super().__init__()
        self.gamma = gamma_

    def forward(self, d_out_fake, d_out_real):
        loss = torch.mean(-self.gamma * torch.log2(d_out_fake) - torch.log2(torch.ones_like(d_out_real) - d_out_real))
        return loss




class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)


class MLP(torch.nn.Module):
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandEdgeSampler(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)
    self.node_list = np.unique(np.concatenate((src_list, dst_list)))

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, edges):
    #总的节点的列表
    num_node = self.node_list.shape[0] #获取了节点数
    num_edge = edges.shape[0]  #获取边数

    negative_edge = edges.copy()  #
    fake_idx = np.random.choice(num_node, num_edge) #随机选取了要产生异常的节点，选取了边数个
    fake_position = np.random.choice(2, num_edge).tolist()  #随机选择异常的是前节点还是后节点
    fake_idx = self.node_list[fake_idx]  # 在选择的位置，全部往前挪了一下
    negative_edge[np.arange(num_edge), fake_position] = fake_idx    
    
    return negative_edge[:,0], negative_edge[:,1]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)
    
class RandEdgeSampler2(object):
  def __init__(self, src_list, dst_list, seed=None):
    self.seed = None
    self.src_list = np.unique(src_list)
    self.dst_list = np.unique(dst_list)

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def sample(self, edges):
    #总的节点的列表
    num_edge = edges.shape[0]  #获取边数
    
    if self.seed is None:
      src_index = np.random.randint(0, len(self.src_list), num_edge)
      dst_index = np.random.randint(0, len(self.dst_list), num_edge)
    else:
      src_index = self.random_state.randint(0, len(self.src_list), num_edge)
      dst_index = self.random_state.randint(0, len(self.dst_list), num_edge)
    return self.src_list[src_index], self.dst_list[dst_index]

  def reset_random_state(self):
    self.random_state = np.random.RandomState(self.seed)  


def get_neighbor_finder(data, uniform, max_node_idx=None):
  max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
  adj_list = [[] for _ in range(max_node_idx + 1)]
  for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
    adj_list[source].append((destination, edge_idx, timestamp))
    adj_list[destination].append((source, edge_idx, timestamp))

  return NeighborFinder(adj_list, uniform=uniform)

class GenFGANLoss(nn.Module):
    def __init__(self, alpha_= 0.1, beta_=15, **kwargs):
        super().__init__()
        self.alpha = alpha_
        self.beta = beta_

    def forward(self, d_out, g_out):
        # calculate loss using the function defined in the paper
        bce = torch.nn.BCELoss()
        EL = bce(d_out, self.alpha * torch.ones_like(d_out))
        mu =  g_out - torch.ones_like(g_out) * g_out.mean(dim=0) 
        
        DL = 0
        for i in range(3):
          DL += 2*torch.norm(mu[:,i], p=2) / (max(g_out[:,i])-min(g_out[:,i]))
          # DL += torch.norm(mu[:,i], p=2) / torch.mean(mu[:,i])
        DL = DL/3
        loss = EL + self.beta * 1/DL
        return loss

class DiscFGANLoss(nn.Module):
    def __init__(self, gamma_=1, **kwargs):
        super().__init__()
        self.gamma = gamma_

    def forward(self, d_out_fake, d_out_real):
        # calculate loss using the function defined in the paper
        loss = torch.mean(-self.gamma * torch.log2(d_out_fake) - torch.log2(torch.ones_like(d_out_real) - d_out_real))
        return loss

class NeighborFinder:
  def __init__(self, adj_list, uniform=False, seed=None):
    self.node_to_neighbors = []
    self.node_to_edge_idxs = []
    self.node_to_edge_timestamps = []

    for neighbors in adj_list:
      # Neighbors is a list of tuples (neighbor, edge_idx, timestamp)
      # We sort the list based on timestamp
      sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
      self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
      self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
      self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

    self.uniform = uniform

    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)

  def find_before(self, src_idx, cut_time):
    """
    Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.

    Returns 3 lists: neighbors, edge_idxs, timestamps

    """
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)

    return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]

  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.

    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert (len(source_nodes) == len(timestamps))

    tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
    # NB! All interactions described in these matrices are sorted in each row by time
    neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
    edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
    edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
      np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

      if len(source_neighbors) > 0 and n_neighbors > 0:
        if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
          sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

          neighbors[i, :] = source_neighbors[sampled_idx]
          edge_times[i, :] = source_edge_times[sampled_idx]
          edge_idxs[i, :] = source_edge_idxs[sampled_idx]

          # re-sort based on time
          pos = edge_times[i, :].argsort()
          neighbors[i, :] = neighbors[i, :][pos]
          edge_times[i, :] = edge_times[i, :][pos]
          edge_idxs[i, :] = edge_idxs[i, :][pos]
        else:
          # Take most recent interactions
          source_edge_times = source_edge_times[-n_neighbors:]
          source_neighbors = source_neighbors[-n_neighbors:]
          source_edge_idxs = source_edge_idxs[-n_neighbors:]

          assert (len(source_neighbors) <= n_neighbors)
          assert (len(source_edge_times) <= n_neighbors)
          assert (len(source_edge_idxs) <= n_neighbors)

          neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
          edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
          edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

    return neighbors, edge_idxs, edge_times