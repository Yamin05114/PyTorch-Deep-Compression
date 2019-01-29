from libs import *


class Compressor(object):
    def __init__(self, model, cuda=False):
        self.model = model
        self.num_layers = 0
        self.num_dropout_layers = 0
        self.dropout_rates = {}

        self.count_layers()
        
        # 初始所有层，每一层是一个None
        self.weight_masks = [None for _ in range(self.num_layers)]
        self.bias_masks = [None for _ in range(self.num_layers)]
        
        self.cuda = cuda
    
    # 计算层数，卷积和fc就算layer
    # drop out不算layer但是记录概率
    def count_layers(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                self.num_layers += 1
            elif isinstance(m, nn.Dropout):
                self.dropout_rates[self.num_dropout_layers] = m.p
                self.num_dropout_layers += 1

    def prune(self):
        '''
        :return: percentage pruned in the network
        '''
        index = 0
        dropout_index = 0

        num_pruned, num_weights = 0, 0

        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                
                # weights 总数
                num = torch.numel(m.weight.data)
                
                # 卷积层 alpha们[0.015 0.2 0.2 0.2]
                if type(m) == nn.Conv2d:
                    if index == 0:
                        alpha = 0.015
                    else:
                        alpha = 0.2
                # fc layers [1,1,1,...,0.25]
                else:
                    if index == self.num_layers - 1:
                        alpha = 0.25
                    else:
                        alpha = 1

                # use a byteTensor to represent the mask and convert it to a floatTensor for multiplication
                # mask删除的门限是weight自己的标准差
                weight_mask = torch.ge(m.weight.data.abs(), alpha * m.weight.data.std()).type('torch.FloatTensor')
                if self.cuda:
                    weight_mask = weight_mask.cuda()
                self.weight_masks[index] = weight_mask  # 存上 weight mask

                bias_mask = torch.ones(m.bias.data.size())
                if self.cuda:
                    bias_mask = bias_mask.cuda()

                # for all kernels in the conv2d layer, if any kernel is all 0, set the bias to 0
                # in the case of linear layers, we search instead for zero rows
                for i in range(bias_mask.size(0)):
                    if len(torch.nonzero(weight_mask[i]).size()) == 0:  # 非零的位置，如果没有biasmask直接清零。
                        bias_mask[i] = 0
                self.bias_masks[index] = bias_mask  # 存上bias mask

                index += 1

                layer_pruned = num - torch.nonzero(weight_mask
                                                  ).size(0)
                logging.info('number pruned in weight of layer %d: %.3f %%' % (index, 100 * (layer_pruned / num)))
                bias_num = torch.numel(bias_mask)
                bias_pruned = bias_num - torch.nonzero(bias_mask).size(0)
                logging.info('number pruned in bias of layer %d: %.3f %%' % (index, 100 * (bias_pruned / bias_num)))

                num_pruned += layer_pruned
                num_weights += num

                m.weight.data *= weight_mask
                m.bias.data *= bias_mask

            elif isinstance(m, nn.Dropout):
                # update the dropout rate
                mask = self.weight_masks[index - 1]
                m.p = self.dropout_rates[dropout_index] * math.sqrt(torch.nonzero(mask).size(0) \
                                             / torch.numel(mask))
                dropout_index += 1
                logging.info("new Dropout rate:", m.p)

        # print(self.weight_masks)
        return num_pruned / num_weights


    def set_grad(self):
        # print(self.weight_masks)
        index = 0
        for m in self.model.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                m.weight.grad.data *= self.weight_masks[index]
                m.bias.grad.data *= self.bias_masks[index]
                index += 1

