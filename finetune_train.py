from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from ImageDataset import ImageDataset
from torchvision import transforms
from DBMNet import Net, pvt_v2_b1
from DBMNet import *
from scipy import stats
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time
from tqdm import tqdm
import logging
import numpy as np

root_dir = '/usr/wsj/dataset/MD_Real_Database'
val_dir = '/usr/wsj/dataset/MD_Uniform_Database'  # 测试集路径
train_txt = '/usr/zhengjia/Work2_wsj/Real_uniform/Real_train_pair.txt'
val_txt = '/usr/zhengjia/Work2_wsj/Real_uniform/uniform_test_pair.txt'
# train_txt = './data/Pair_first_train_name_mos.txt'
# 获取当前文件的绝对路径
Absolute_path = os.getcwd()


# 格式化日期
def time_to_time(one_time, mode='str'):
    one_time = round(one_time)
    m, s = divmod(one_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    t = [d, h, m, s]
    if mode == 'list':
        return t
    if mode == 'str':
        time_str = str()
        str_list = ['d ', ':', ':', ' ']
        for i in range(len(t)):
            if t[i] != 0:
                time_str += str(t[i]) + str_list[i]
        return time_str


# 排序loss
# def rank_loss(d, y, num_images, eps=1e-6, norm_num=True):
#     loss = torch.zeros(1, device=device)
#
#     if num_images < 2:
#         return loss
#
#     dp = torch.abs(d)
#
#     combinations = torch.combinations(torch.arange(num_images), 2)
#     combinations_count = max(1, len(combinations))
#
#     for i, j in combinations:
#         rl = torch.clamp_min(-(y[i] - y[j]) * (d[i] - d[j]) / (torch.abs(y[i] - y[j]) + eps), min=0)
#         loss += rl / max(dp[i], dp[j])  # normalize by maximum value
#
#     if norm_num:
#         loss = loss / combinations_count  # mean
#
#     return loss

# 模型测试
def IQAPerformance(dataloader, model, device, n=13):
    '''

    :param data_loader:
    :param model:
    :param device:
    :param n:   算法总个数
    :return:
    '''
    pscores = []
    tscores = []
    Srcc = []
    Plcc = []
    Krocc = []
    Rmse = []
    with torch.no_grad():
        for step, data in enumerate(dataloader):
            # Data
            degradeImage, restoreImage, D_mos = data
            D_mos = D_mos.view(degradeImage.size()[0], -1)
            degradeImage = degradeImage.to(device, dtype=torch.float32)
            restoreImage = restoreImage.to(device, dtype=torch.float32)
            D_mos = D_mos.to(device, dtype=torch.float32)

            inputs = [degradeImage, restoreImage]
            # 每个阶段都预测一组值
            q = model(inputs)
            pscores = pscores + q.squeeze(1).cpu().tolist()
            tscores = tscores + D_mos.squeeze(1).cpu().tolist()

        pscores = np.array(pscores, dtype='float32').reshape(-1, n)
        tscores = np.array(tscores, dtype='float32').reshape(-1, n)
        logging.info(f'***************{pscores.shape[1]}个算法为一组***************')
        for i in range(pscores.shape[0]):
            test_srcc, _ = stats.spearmanr(pscores[i], tscores[i])
            test_plcc, _ = stats.pearsonr(pscores[i], tscores[i])
            test_krocc, _ = stats.stats.kendalltau(pscores[i], tscores[i])
            test_RMSE = np.sqrt(((pscores[i] - tscores[i]) ** 2).mean())
            Srcc = Srcc + [test_srcc]
            Plcc = Plcc + [test_plcc]
            Krocc = Krocc + [test_krocc]
            Rmse = Rmse + [test_RMSE]
    return np.mean(Srcc), np.mean(Plcc), np.mean(Krocc), np.mean(Rmse)


def train(model,
          device,
          start_epoch=0,
          epochs=5,
          batch_size=2,
          checkpoint_dir=None,
          writer=None,
          Image_size=None,
          Test_size=None,
          ):
    # 训练数据预处理
    train_transform = transforms.Compose([
        transforms.Resize(Image_size),  # 将图片设置为（512，512）的尺寸
        transforms.ToTensor(),  # 图片转张量，同时归一化 0-255 ---》 0-1
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 减均值，除方差(-1,1)
    ])

    # 测试数据预处理 0.5, 0.5, 0.5
    valid_transform = transforms.Compose([
        transforms.Resize(Test_size),  # 将图片设置为（512，512）的尺寸
        transforms.ToTensor(),  # 图片转张量，同时归一化0-255 ---》 0-1
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 减均值，除方差(-1,1)
    ])

    # 读取数据
    train_data = ImageDataset(root_dir, train_txt, train_transform)
    val_data = ImageDataset(val_dir, val_txt, valid_transform)
    n_train = len(train_data)  # 训练总样本数
    n_valid = len(val_data)  # 测试总样本数
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
    batchs_num = len(train_dataloader)  # 总的训练批次数
    iteration = 0  # 总迭代次数
    max_Srcc = -2  # 记录最好的SRcc评分

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Training size:   {n_train}
        Validation size: {n_valid}
        Checkpoints:     {checkpoint_dir}
        Device:          {device}
        start_epoch:     {start_epoch}
        Image_size:      {Image_size}
        Test_size:      {Test_size}
    ''')

    start_time = time.time()  # 开始时间

    # 定义loss，优化器
    loss_fn = nn.MSELoss(reduction='mean').to(device)
    Pvt_params = list(map(id, model.Pvt.parameters()))
    base_params = filter(lambda p: id(p) not in Pvt_params, model.parameters())
    optimizer = torch.optim.AdamW([
        {'params': model.Pvt.parameters()},
        {'params': base_params, 'lr': 1e-4}], lr=1e-5, weight_decay=1e-2)
    # lambda1 = lambda epoch: 1 / (10 ** (epoch // 10))
    # lambda1 = lambda epoch: 0.5**(epoch//5)
    # lambda2 = lambda epoch: 1 / (10 ** (epoch // 10))
    # lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

    for epoch in range(start_epoch, epochs):
        # 计算进度条时间
        run_time = time.time() - start_time
        remain_time = run_time * (epochs - epoch) / max(1, (epoch - start_epoch))
        run_time = time_to_time(run_time)
        remain_time = time_to_time(remain_time)

        total_loss = 0
        epoch_loss = 0
        pre_score = []
        tar_score = []
        # 训练步骤开始
        model.train()
        with tqdm(total=batchs_num, desc=f'Epoch {epoch + 1}/{start_epoch + epochs} {run_time}~{remain_time}',
                  unit='Batch', ncols=120) as pbar:
            for step, data in enumerate(train_dataloader):
                degradeImage, restoreImage, D_mos = data
                D_mos = D_mos.view(degradeImage.size()[0], -1)
                # 移到GPU
                degradeImage = degradeImage.to(device, dtype=torch.float32)
                restoreImage = restoreImage.to(device, dtype=torch.float32)
                D_mos = D_mos.to(device, dtype=torch.float32)

                inputs = [degradeImage, restoreImage]
                q = model(inputs)
                # print(D_mos.shape)
                # print(q.shape)
                loss = loss_fn(q, D_mos)
                total_loss = total_loss + loss.item()  # 批次总loss

                pre_score = pre_score + q.squeeze(1).cpu().tolist()
                tar_score = tar_score + D_mos.squeeze(1).cpu().tolist()

                # 优化器优化模型
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iteration = iteration + 1
                if (iteration) % 100 == 0:
                    # print("\n训练次数：{}      Loss：{}".format(iteration, loss.item()))
                    writer.add_scalar("iteration_loss", loss.item(), iteration)
                    logging.info(
                        f'训练次数：{iteration}       Loss：{round(loss.item(), 6)}')
                pbar.update()
        # lr_scheduler.step()
        # print('epoch: ', epoch, 'lr: ', lr_scheduler.get_lr())

        train_srcc, _ = stats.spearmanr(pre_score, tar_score)
        epoch_loss = total_loss / batchs_num

        writer.add_scalar("epoch_loss", epoch_loss, epoch)
        print("训练epoch：{}       训练SRCC：{}       EpochLoss：{}".format(epoch + 1, train_srcc, epoch_loss))
        logging.info(f'训练epoch：{epoch + 1}       训练SRCC：{train_srcc}')
        logging.info(f'EpochLoss：{epoch_loss}')

        # 保存最后一个epoch的模型
        final_model_name = os.path.join(checkpoint_dir, 'final_model.pth')
        torch.save({
            'final_model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, final_model_name)
        logging.info(f'Checkpoint final model saved! epoch: {epoch + 1} ')
        logging.info('-' * 88)

        # 保存每一次模型
        # if (epoch + 1) % 1 == 0:
        #     model_name = os.path.join(checkpoint_dir, 'epoch_{0}model.pth'.format(epoch + 1))
        #     torch.save(model.state_dict(), model_name)
        #     torch.save({
        #                 'model': model.state_dict(),
        #                 'optimizer': optimizer.state_dict(),
        #             }, model_name)

        # 测试步骤开始
        model.eval()
        logging.info("测试结果：")
        test_srcc, test_plcc, test_krocc, test_RMSE = IQAPerformance(val_dataloader, model, device, n=13)
        logging.info(f'测试集SRCC：{test_srcc}  测试集PLCC：{test_plcc}')
        logging.info(f'测试集KROCC：{test_krocc}  测试集RMSE：{test_RMSE}')
        logging.info('-' * 88)

        if test_srcc > max_Srcc:
            max_Srcc = max(max_Srcc, test_srcc)
            # 保存测试效果最好的模型
            best_model_name = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'best_model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, best_model_name)
            logging.info(f'Checkpoint best model saved! epoch: {epoch + 1}')
        logging.info("\n\n")

    end = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        end // 60, end % 60))


if __name__ == '__main__':
    # 获取当前时间
    time_now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    # 定义日志路径
    log_dir = './logs'
    try:
        os.mkdir(log_dir)
    except OSError:
        pass
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        filename=os.path.join(log_dir, "{}.log".format(time_now)),
                        filemode='w')
    if os.path.exists(log_dir):
        logging.info("Created log_dir directory:" + str(log_dir))

    # 定义保存模型的路径
    checkpoint_dir = os.path.join(Absolute_path, "save_models", time_now)
    try:
        os.makedirs(checkpoint_dir)
        logging.info('Created checkpoint directory:' + str(checkpoint_dir))
    except OSError:
        pass
    # 可视化日志路径
    runs_dir = os.path.join(Absolute_path, "runs", time_now)
    logging.info('Created runs_dir directory:' + str(runs_dir))
    writer = SummaryWriter(runs_dir)
    # 打印训练集和测试集路径
    logging.info('训练路径：' + str(root_dir))
    logging.info('测试路径：' + str(val_dir))
    logging.info('训练集：' + str(train_txt))
    logging.info('测试集：' + str(val_txt))
    # 定义GPU设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 定义PVT模型
    pvt = pvt_v2_b1()
    checkpoint = "./pvt_v2_b1.pth"
    pvt.load_state_dict(
        torch.load(checkpoint,
                   map_location=device))

    # [32, 64, 160, 256]   [64, 128, 320, 512]
    model = Net(pvt=pvt, dims=[64, 128, 320, 512])
    logging.info('加载预训练模型:' + str(checkpoint))
    model.to(device)
    logging.info(f'''model:
                    {model}''')

    train(model=model,
          device=device,
          start_epoch=0,
          epochs=80,
          batch_size=16,
          checkpoint_dir=checkpoint_dir,
          writer=writer,
          Image_size=(512, 512),
          Test_size=(512, 512)
          )
    writer.close()
    pass
