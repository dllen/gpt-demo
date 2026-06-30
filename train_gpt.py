#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文 GPT 模型 —— 纯 NumPy 从零实现
====================================
一个可运行的 GPT (Decoder-Only Transformer) 训练脚本，不依赖 PyTorch/TensorFlow/JAX。
支持中文语料训练、权重共享、Adam 优化器（带 warmup 与 cosine 衰减）、梯度裁剪、
模型保存/加载、文本生成（支持 temperature 与 top-k 采样）。

直接运行: python train_gpt.py
"""

import numpy as np
import os
import sys
import time
from collections import Counter

try:
    import requests
except ImportError:
    requests = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ===========================================================================
# 模块 0: 全局超参数
# ===========================================================================
VOCAB_SIZE    = 5000
D_MODEL       = 64
N_HEADS       = 4
HEAD_DIM      = D_MODEL // N_HEADS   # 16
N_LAYERS      = 4
D_FF          = 256
MAX_SEQ_LEN   = 128
DROPOUT       = 0.1
BATCH_SIZE    = 64
LEARNING_RATE = 3e-3
NUM_EPOCHS    = 60
CLIP_GRAD     = 1.0
WARMUP_STEPS  = 500
PRINT_EVERY   = 100
SAVE_EVERY    = 2000
MODEL_SAVE_PATH = "gpt_chinese.npz"
CORPUS_CACHE_PATH = "corpus.txt"


# ===========================================================================
# 模块 1: 数据下载与处理
# ===========================================================================

def download_corpus():
    """
    获取中文语料。
    优先使用本地缓存 corpus.txt；若无缓存则从网络下载全唐诗并保存到本地。
    返回合并后的中文文本字符串。
    """
    if os.path.exists(CORPUS_CACHE_PATH):
        with open(CORPUS_CACHE_PATH, 'r', encoding='utf-8') as f:
            corpus = f.read()
        print(f"[数据] 从本地缓存加载语料: {CORPUS_CACHE_PATH} ({len(corpus)} 字符)")
        return corpus

    print("[数据] 本地缓存不存在，开始从网络下载语料...")

    # 内置 fallback 语料: 经典古诗词 + 常用文本片段
    fallback_corpus = """
    床前明月光疑是地上霜举头望明月低头思故乡
    春眠不觉晓处处闻啼鸟夜来风雨声花落知多少
    白日依山尽黄河入海流欲穷千里目更上一层楼
    红豆生南国春来发几枝愿君多采撷此物最相思
    锄禾日当午汗滴禾下土谁知盘中餐粒粒皆辛苦
    离离原上草一岁一枯荣野火烧不尽春风吹又生
    好雨知时节当春乃发生随风潜入夜润物细无声
    两个黄鹂鸣翠柳一行白鹭上青天窗含西岭千秋雪门泊东吴万里船
    朝辞白帝彩云间千里江陵一日还两岸猿声啼不住轻舟已过万重山
    日照香炉生紫烟遥看瀑布挂前川飞流直下三千尺疑是银河落九天
    千山鸟飞绝万径人踪灭孤舟蓑笠翁独钓寒江雪
    故人西辞黄鹤楼烟花三月下扬州孤帆远影碧空尽唯见长江天际流
    天门中断楚江开碧水东流至此回两岸青山相对出孤帆一片日边来
    葡萄美酒夜光杯欲饮琵琶马上催醉卧沙场君莫笑古来征战几人回
    独在异乡为异客每逢佳节倍思亲遥知兄弟登高处遍插茱萸少一人
    空山不见人但闻人语响返景入深林复照青苔上
    松下问童子言师采药去只在此山中云深不知处
    春种一粒粟秋收万颗子四海无农夫犹饿死
    远上寒山石径斜白云深处有人家停车坐爱枫林晚霜叶红于二月花
    月落乌啼霜满天江枫渔火对愁眠姑苏城外寒山寺夜半钟声到客船
    海上生明月天涯共此时情人怨遥夜竟夕起相思
    人生若只如初见何事秋风悲画屏等闲变却故人心却道故人心易变
    大江东去浪淘尽千古风流人物故垒西边人道是三国周郎赤壁
    明月几时有把酒问青天不知天上宫阙今夕是何年
    枯藤老树昏鸦小桥流水人家古道西风瘦马夕阳西下断肠人在天涯
    寻寻觅觅冷冷清清凄凄惨惨戚戚乍暖还寒时候最难将息
    十年生死两茫茫不思量自难忘千里孤坟无处话多情应笑我早生华发
    人生得意须尽欢莫使金樽空对月天生我材必有用千金散尽还复来
    君不见黄河之水天上来奔流到海不复回君不见高堂明镜悲白发朝如青丝暮成雪
    国破山河在城春草木深感时花溅泪恨别鸟惊心烽火连三月家书抵万金
    安得广厦千万间大庇天下寒士俱欢颜风雨不动安如山呜呼何时眼前突兀见此屋
    大江来从万山中山势尽与江流东孤风曹公乔安足雄霸业
    秦时明月汉时关万里长征人未还但使龙城飞将在不教胡马度阴山
    黄河远上白云间一片孤城万仞山羌笛何须怨杨柳春风不度玉门关
    谁家玉笛暗飞声散入春风满洛城此夜曲中闻折柳何人不起故园情
    渭城朝雨浥轻尘客舍青青柳色新劝君更尽一杯酒西出阳关无故人
    远芳侵古道晴翠接荒城又送王孙去萋萋满别情
    凉州词葡萄美酒夜光杯边塞凄凉塞北风
    出塞秦时明月汉时关征人戍边思故乡
    塞下曲月黑雁飞高单于夜遁逃欲将轻骑逐大雪满弓刀
    鹿柴空山不见人但闻人语响返景入深林复照青苔上
    相思红豆生南国春来发几枝愿君多采撷此物最相思
    杂诗君自故乡来应知故乡事来日绮窗前寒梅著花未
    送别山中相送罢日暮掩柴扉春草明年绿王孙归不归
    终南中岁颇好道晚家南山陲兴来每独往胜事空自知
    行到水穷处坐看云起时偶然值林叟谈笑无还期
    鹿柴空山不见人但闻人语响返景入深林复照青苔上
    竹里馆独坐幽篁里弹琴复长啸深林人不知明月来相照
    送别下马饮君酒问君何所之君言不得意归卧南山陿
    春晓春眠不觉晓处处闻啼鸟夜来风雨声花落知多少
    宿建德江移舟泊烟渚日暮客愁新野旷天低树江清月近人
    江雪千山鸟飞绝万径人踪灭孤舟蓑笠翁独钓寒江雪
    登鹳雀楼白日依山尽黄河入海流欲穷千里目更上一层楼
    静夜思床前明月光疑是地上霜举头望明月低头思故乡
    望庐山瀑布日照香炉生紫烟遥看瀑布挂前川飞流直下三千尺疑是银河落九天
    赠汪伦李白乘舟将欲行忽闻岸上踏歌声桃花潭水深千尺不及汪伦送我情
    黄鹤楼送孟浩然之广陵故人西辞黄鹤楼烟花三月下扬州孤帆远影碧空尽唯见长江天际流
    早发白帝城朝辞白帝彩云间千里江陵一日还两岸猿声啼不住轻舟已过万重山
    望天门山天门中断楚江开碧水东流至此回两岸青山相对出孤帆一片日边来
    绝句两个黄鹂鸣翠柳一行白鹭上青天窗含西岭千秋雪门泊东吴万里船
    春夜喜雨好雨知时节当春乃发生随风潜入夜润物细无声野径云俱黑江船火独明晓看红湿处花重锦官城
    枫桥夜泊月落乌啼霜满天江枫渔火对愁眠姑苏城外寒山寺夜半钟声到客船
    山行远上寒山石径斜白云深处有人家停车坐爱枫林晚霜叶红于二月花
    泊船瓜洲京口瓜洲一水间钟山只隔数重山春风又绿江南岸明月何时照我还
    饮湖上初晴后雨水光潋滟晴方好山色空蒙雨亦奇欲把西湖比西子淡妆浓抹总相宜
    惠崇春江晚景竹外桃花三两枝春江水暖鸭先知蒌蒿满地芦芽短正是河豚欲上时
    题西林壁横看成岭侧成峰远近高低各不同不识庐山真面目只缘身在此山中
    示儿死去元知万事空但悲不见九州同王师北定中原日家祭无忘告乃翁
    小池泉眼无声惜细流树阴照水爱晴柔小荷才露尖尖角早有蜻蜓立上头
    晓出净慈寺送林子方毕竟西湖六月中风光不与四时同接天莲叶无穷碧映日荷花别样红
    春日胜日寻芳泗水滨无边光景一时等新闲识得东风面万紫千红总是春
    观书有感半亩方塘一鉴开天光云影共徘徊问渠那得清如许为有源头活水来
    石灰吟千锤万凿出深山烈火焚烧若等闲粉骨碎身浑不怕要留清白在人间
    竹石咬定青山不放松立根原在破岩中千磨万击还坚劲任尔东西南北风
    己亥杂诗浩荡离愁白日斜吟鞭东指即天涯落红不是无情物化作春泥更护花
    长歌行百川东到海何时复西归少壮不努力老大徒伤悲
    七步诗煮豆燃豆萁豆在釜中泣本是同根生相煎何太急
    敕勒歌敕勒川阴山下天似穹庐笼盖四野天苍苍野茫茫风吹草低见牛羊
    咏鹅鹅鹅鹅曲项向天歌白毛浮绿水红掌拨清波
    回乡偶书少小离家老大回乡音无改鬓毛衰儿童相见不相识笑问客从何处来
    咏柳碧玉妆成一树高万条垂下绿丝绦不知细叶谁裁出二月春风似剪刀
    凉州词黄河远上白云间一片孤城万仞山羌笛何须怨杨柳春风不度玉门关
    登鹳雀楼白日依山尽黄河入海流欲穷千里目更上一层楼
    凉州词葡萄美酒夜光杯欲饮琵琶马上催醉卧沙场君莫笑古来征战几人回
    出塞秦时明月汉时关万里长征人未还但使龙城飞将在不教胡马度阴山
    芙蓉楼送寒雨连江夜入吴平明送客楚山孤洛阳亲友如相问一片冰心在玉壶
    人日思归入才七日离家已二年人归落雁后思发在花前
    鸟鸣涧人闲桂花落夜静春山空月出惊山鸟时鸣春涧中
    九月九日忆山东兄弟独在异乡为异客每逢佳节倍思亲遥知兄弟登高处遍插茱萸少一人
    静夜思床前明月光疑是地上霜举头望明月低头思故乡
    望庐山瀑布日照香炉生紫烟遥看瀑布挂前川飞流直下三千尺疑是银河落九天
    早发白帝城朝辞白帝彩云间千里江陵一日还两岸猿声啼不住轻舟已过万重行
    送元二使安西渭城朝雨浥轻尘客舍青青柳色新劝君更尽一杯酒西出阳关无故人
    别董大千里黄云白日曛北风吹雁雪纷纷莫愁前路无知己天下谁人不识君
    江南春千里莺啼绿映红水村山郭酒旗风南朝四百八十寺多少楼台烟雨中
    夏日绝句生当作人杰死亦为鬼雄至今思项羽不肯过江东
    元日爆竹声中一岁除春风送暖入屠苏千门万户曈曈日总把新桃换旧符
    泊船瓜洲京口瓜洲一水间钟山只隔数重山春风又绿江南岸明月何时照我还
    梅花墙角数枝梅凌寒独自开遥知不是雪为有暗香来
    题临安邸山外青山楼外楼西湖歌舞几时休暖风熏得游人醉直把杭州作汴州
    游园不值应怜屐齿印苍苔小扣柴扉久不开春色满园关不住一枝红杏出墙来
    墨梅吾家洗砚池头树个个花开淡墨痕不要人夸好颜色只留清气满乾坤
    劝学三更灯火五更鸡正是男儿读书时黑发不知勤学早白首方悔读书迟
    悯农春种一粒粟秋收万颗子四海无闲田农夫犹饿死
    悯农锄禾日当午汗滴禾下土谁知盘中餐粒粒皆辛苦
    游子吟慈母手中线游子身上衣临行密密缝意恐迟迟归谁言寸草心报得三春晖
    清明清明时节雨纷纷路上行人欲断魂借问酒家何处有牧童遥指杏花村
    端午粽香飘五月龙舟竞渡长江
    中秋月圆人团圆举杯邀明月
    春节爆竹声声辞旧岁梅花朵朵迎新年
    人工智能是计算机科学的一个分支它企图了解智能的实质并生产出一种新的能以人类智能相似的方式做出反应的智能机器
    机器学习是人工智能的一个子领域它使用算法让计算机从数据中学习规律从而能够对新数据进行预测或决策
    深度学习是机器学习的一个分支它使用多层神经网络来学习数据的复杂特征表示
    自然语言处理是人工智能的一个重要方向它研究如何让计算机理解和生成人类语言
    深度学习在图像识别语音识别自然语言处理等领域取得了巨大的成功
    神经网络由大量的神经元相互连接构成每个神经元接收输入进行加权求和然后通过激活函数产生输出
    反向传播算法是训练神经网络的核心算法它通过计算损失函数对每个权重的梯度然后使用梯度下降来更新权重
    梯度下降是优化神经网络的基本方法它沿着梯度的反方向更新参数以最小化损失函数
    卷积神经网络是一种专门用于处理具有网格结构数据的神经网络它广泛应用于图像处理领域
    循环神经网络是一种专门用于处理序列数据的神经网络它具有记忆能力可以记住之前的输入信息
    注意力机制是一种让模型能够关注输入数据中重要部分的技术它广泛应用于自然语言处理领域
    编码器解码器架构是一种常用的序列到序列模型编码器将输入序列编码为一个固定长度的向量解码器将该向量解码为输出序列
    词嵌入是一种将词语映射到低维稠密向量空间的技术它能够捕捉词语之间的语义关系
    生成式预训练变换器是一种基于变换器架构的预训练语言模型它在自然语言生成任务中表现出色
    预训练模型通过在大规模语料上无监督学习获得通用的语言表示能力然后在特定任务上进行微调
    机器学习模型的性能取决于多个因素包括特征选择模型结构训练数据质量和超参数设置
    过拟合是机器学习中常见的问题指模型在训练数据上表现很好但在测试数据上表现较差
    正则化是防止过拟合的技术它通过在损失函数中加入惩罚项来限制模型的复杂度
    交叉验证是一种评估模型性能的方法它将数据分为多个子集轮流用其中一个作为测试集
    特征工程是机器学习中的关键步骤它通过选择转换和创建合适的特征来提高模型性能
    数据预处理包括数据清洗缺失值处理特征缩放编码分类变量等步骤
    模型评估指标包括准确率精确率召回率F1分数等
    梯度消失和梯度消失是深度神经网络训练中的常见问题它会导致浅层网络参数难以更新
    批量归一化是一种常用的正则化技术通过对每一层的输入做归一化来加速训练并提高模型稳定性
    残差连接通过将层的输入直接加到输出上解决了深层网络训练困难的问题
    多头注意力机制通过并行计算多组注意力来捕捉数据的不同表示子空间
    位置编码在变换器中用于为模型提供序列中每个位置的信息因为变换器本身不具备位置感知能力
    层归一化对每一层的激活值进行归一化有助于稳定深层网络的训练
    激活函数引入非线性因素使神经网络能够逼近复杂的函数
    损失函数衡量模型预测值与真实值之间的差异是优化算法的目标函数
    学习率控制参数更新的步长是训练神经网络中最重要的超参数之一
    批量大小影响梯度估计的准确性和训练的内存占用
    训练轮数决定整个数据集被训练多少次过多会导致过拟合过少会导致欠拟合
    """
    corpus = fallback_corpus.strip()

    urls = [
        f"https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/全唐诗/poet.tang.{i}.json"
        for i in range(0, 5000, 1000)
    ]

    downloaded_parts = []
    if requests is not None:
        for url in urls:
            try:
                print(f"[数据] 尝试下载: {url}")
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                for item in data:
                    title = "".join(c for c in item.get("title", "") if '\u4e00' <= c <= '\u9fff')
                    author = "".join(c for c in item.get("author", "") if '\u4e00' <= c <= '\u9fff')
                    paragraphs = item.get("paragraphs", [])
                    for line in paragraphs:
                        line_clean = "".join(c for c in line if '\u4e00' <= c <= '\u9fff')
                        if line_clean:
                            downloaded_parts.append(line_clean)
                    if title:
                        downloaded_parts.append(title)
                    if author:
                        downloaded_parts.append(author)
                print(f"[数据] 从 {url} 获取了 {len(data)} 首诗词")
            except Exception as e:
                print(f"[数据] 下载失败: {e}")
    else:
        print("[数据] requests 库不可用，使用内置语料")

    if downloaded_parts:
        downloaded_text = "".join(downloaded_parts)
        if len(downloaded_text) > 5000:
            corpus = downloaded_text + corpus

    # 保存到本地缓存
    with open(CORPUS_CACHE_PATH, 'w', encoding='utf-8') as f:
        f.write(corpus)
    print(f"[数据] 语料已保存到本地缓存: {CORPUS_CACHE_PATH} ({len(corpus)} 字符)")

    return corpus


def build_vocab(text, vocab_size):
    """
    根据字符频率构建词汇表。
    保留最常见的 (vocab_size - 4) 个字符,加上 4 个特殊 token:
    <PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3
    返回 char2idx 字典, idx2char 字典
    """
    counter = Counter(text)
    # 取频率最高的 vocab_size - 4 个字符
    most_common = counter.most_common(vocab_size - 4)
    idx2char = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    for char, _ in most_common:
        idx2char.append(char)
    char2idx = {ch: i for i, ch in enumerate(idx2char)}
    print(f"[词汇表] 大小: {len(idx2char)}, 最常见的字符: {most_common[:10]}")
    return char2idx, idx2char


def prepare_data(text, char2idx, seq_len):
    """
    将文本转为 token id 列表,按 seq_len + 1 长度滑动窗口切分。
    构建 (input, target) 对: input = tokens[:-1], target = tokens[1:]
    返回 inputs (NumPy array, shape: (num_samples, seq_len))
          targets  (NumPy array, shape: (num_samples, seq_len))
    """
    # 将每个字符转为 token id,未知字符映射为 <UNK>=1
    token_ids = [char2idx.get(ch, 1) for ch in text]

    inputs = []
    targets = []
    stride = max(1, seq_len // 2)  # 滑动窗口步长为 seq_len/2
    for i in range(0, len(token_ids) - seq_len, stride):
        chunk = token_ids[i:i + seq_len + 1]
        if len(chunk) < seq_len + 1:
            break
        inputs.append(chunk[:-1])
        targets.append(chunk[1:])

    inputs = np.array(inputs, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)
    print(f"[数据] 样本数: {inputs.shape[0]}, 序列长度: {seq_len}")
    return inputs, targets


def get_batch(inputs, targets, batch_size):
    """
    随机采样一个 batch。
    返回 (batch_input, batch_target), shape 均为 (batch_size, seq_len)
    """
    n = inputs.shape[0]
    indices = np.random.randint(0, n, size=batch_size)
    return inputs[indices], targets[indices]


# ===========================================================================
# 模块 2: 模型组件
# ===========================================================================

def xavier_init(fan_in, fan_out):
    """
    Xavier (Glorot) 初始化。
    scale = sqrt(2 / (fan_in + fan_out)),从正态分布采样后乘以 scale。
    方差比保持恒定有利于信号在深度网络中稳定传播。
    """
    scale = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out).astype(np.float32) * scale


def init_model():
    """
    初始化所有模型参数,返回参数字典。
    关键设计: token_embedding 与 LM Head 共享权重(通过转置实现),
    这减少了参数量并提高了 token 表示的利用率。
    """
    params = {}
    rng = np.random.default_rng(42)  # 固定种子确保可复现

    # Token 嵌入: (VOCAB_SIZE, D_MODEL)
    params['token_embedding'] = rng.standard_normal((VOCAB_SIZE, D_MODEL)).astype(np.float32) * 0.02

    # 位置嵌入: (MAX_SEQ_LEN, D_MODEL)
    params['position_embedding'] = rng.standard_normal((MAX_SEQ_LEN, D_MODEL)).astype(np.float32) * 0.02

    # 每个 Transformer 层的参数
    for i in range(N_LAYERS):
        prefix = f'block_{i}'
        # Pre-LayerNorm 1
        params[f'{prefix}_ln1_gamma'] = np.ones(D_MODEL, dtype=np.float32)
        params[f'{prefix}_ln1_beta'] = np.zeros(D_MODEL, dtype=np.float32)
        # 多头注意力的 Q, K, V, O 投影矩阵
        params[f'{prefix}_W_q'] = xavier_init(D_MODEL, D_MODEL)
        params[f'{prefix}_W_k'] = xavier_init(D_MODEL, D_MODEL)
        params[f'{prefix}_W_v'] = xavier_init(D_MODEL, D_MODEL)
        params[f'{prefix}_W_o'] = xavier_init(D_MODEL, D_MODEL)
        # Pre-LayerNorm 2
        params[f'{prefix}_ln2_gamma'] = np.ones(D_MODEL, dtype=np.float32)
        params[f'{prefix}_ln2_beta'] = np.zeros(D_MODEL, dtype=np.float32)
        # 前馈网络: D_MODEL → D_FF → D_MODEL
        params[f'{prefix}_W1'] = xavier_init(D_MODEL, D_FF)
        params[f'{prefix}_b1'] = np.zeros(D_FF, dtype=np.float32)
        params[f'{prefix}_W2'] = xavier_init(D_FF, D_MODEL)
        params[f'{prefix}_b2'] = np.zeros(D_MODEL, dtype=np.float32)

    # 最终的 LayerNorm
    params['ln_f_gamma'] = np.ones(D_MODEL, dtype=np.float32)
    params['ln_f_beta'] = np.zeros(D_MODEL, dtype=np.float32)

    # 计算总参数量
    total = sum(v.size for v in params.values())
    print(f"[模型] 参数总数: {total:,} (~{total / 1e6:.2f}M)")
    return params


def causal_mask(seq_len):
    """
    创建因果掩码(上三角为 -1e9,下三角为 0)。
    在注意力计算中,将上三角位置(对应未来 token)设为极大负值,
    使其 softmax 后趋近于 0,注意力无法看到当前位置之后的内容。
    """
    mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1)
    return mask * -1e9


def layer_norm(x, gamma, beta, eps=1e-5):
    """
    对最后一维做 Layer Normalization。
    对每个样本独立归一化(与 BatchNorm 不同),适合自回归模型。
    同时返回 cache 供反向传播使用。
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    cache = (x, x_norm, mean, var, gamma, eps)
    return out, cache


def layer_norm_backward(dout, cache):
    """
    计算 LayerNorm 的反向传播。
    推导: x_norm = (x - mean) / sqrt(var + eps)
    out = gamma * x_norm + beta
    需要分别计算对 x, gamma, beta 的梯度。
    """
    x, x_norm, mean, var, gamma, eps = cache
    N = x.shape[-1]

    dgamma = np.sum(dout * x_norm, axis=(0, 1) if dout.ndim == 3 else 0)
    dbeta = np.sum(dout, axis=(0, 1) if dout.ndim == 3 else 0)

    # 对 x 的完整梯度推导(考虑 mean 和 var 对 x 的依赖)
    dx_norm = dout * gamma
    std_inv = 1.0 / np.sqrt(var + eps)
    dx = std_inv * (dx_norm - dx_norm.mean(axis=-1, keepdims=True)
                    - x_norm * np.mean(dx_norm * x_norm, axis=-1, keepdims=True))
    return dx, dgamma, dbeta


def softmax(x, axis=-1):
    """
    数值稳定的 softmax: 先减去最大值再求指数,避免指数爆炸。
    """
    x_max = x.max(axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=axis, keepdims=True)


def multi_head_attention_forward(x, W_q, W_k, W_v, W_o, mask, training=True):
    """
    多头注意力的前向传播。
    1. 线性投影 Q, K, V
    2. 拆分成多头并转置
    3. 计算注意力分数并应用因果掩码
    4. softmax 归一化
    5. dropout (训练时)
    6. 加权求和得到 context
    7. 合并多头并通过输出投影
    返回 output 和 cache(供反向传播使用)。
    """
    batch, seq_len, d_model = x.shape
    head_dim = d_model // N_HEADS

    # 线性投影
    Q = x @ W_q  # (batch, seq_len, d_model)
    K = x @ W_k
    V = x @ W_v

    # 拆分成多头: (batch, seq_len, d_model) → (batch, N_HEADS, seq_len, head_dim)
    Q = Q.reshape(batch, seq_len, N_HEADS, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, N_HEADS, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, N_HEADS, head_dim).transpose(0, 2, 1, 3)

    # 注意力分数: Q @ K^T / sqrt(head_dim)
    scale = np.sqrt(head_dim)
    scores = Q @ K.transpose(0, 1, 3, 2) / scale  # (batch, N_HEADS, seq_len, seq_len)

    # 应用因果掩码: 上三角设为 -1e9
    scores = scores + mask[np.newaxis, np.newaxis, :, :]

    # softmax 归一化
    attn_weights = softmax(scores, axis=-1)  # (batch, N_HEADS, seq_len, seq_len)

    # Dropout (训练时)
    dropout_mask = None
    if training and DROPOUT > 0:
        # 生成 dropout mask: 以概率 1-DROPOUT 保留,并做 inverted dropout 缩放
        dropout_mask = (np.random.rand(*attn_weights.shape) > DROPOUT).astype(np.float32)
        attn_weights = attn_weights * dropout_mask / (1.0 - DROPOUT)

    # 加权求和
    context = attn_weights @ V  # (batch, N_HEADS, seq_len, head_dim)

    # 合并多头: (batch, N_HEADS, seq_len, head_dim) → (batch, seq_len, d_model)
    context = context.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)

    # 输出投影
    output = context @ W_o

    cache = (x, Q, K, V, attn_weights, W_q, W_k, W_v, W_o, mask, scale, dropout_mask)
    return output, cache


def multi_head_attention_backward(dout, cache):
    """
    多头注意力的反向传播。
    自注意力是模型中最复杂的部分,需要仔细推导每一步的梯度。
    输入 dout shape: (batch, seq_len, d_model)
    返回 (dx, dW_q, dW_k, dW_v, dW_o)
    """
    x, Q, K, V, attn_weights, W_q, W_k, W_v, W_o, mask, scale, dropout_mask = cache
    batch, seq_len, d_model = x.shape
    head_dim = d_model // N_HEADS

    # ---- 1. 反向通过输出投影: output = context_merged @ W_o ----
    # context_merged: (batch, seq_len, d_model),由多头 context 合并而来
    context = attn_weights @ V  # (batch, N_HEADS, seq_len, head_dim)
    context_merged = context.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)

    dcontext_merged = dout @ W_o.T  # (batch, seq_len, d_model)
    # dW_o = context_merged^T @ dout,将所有样本展平后计算
    dW_o = context_merged.reshape(-1, d_model).T @ dout.reshape(-1, d_model)  # (d_model, d_model)

    # ---- 2. 反向通过 reshape/transpose: 恢复 dcontext ----
    dcontext = dcontext_merged.reshape(batch, seq_len, N_HEADS, head_dim).transpose(0, 2, 1, 3)
    # dcontext shape: (batch, N_HEADS, seq_len, head_dim)

    # ---- 3. 反向通过 context = attn_weights @ V ----
    # context[b,h,s,d] = sum_t attn_weights[b,h,s,t] * V[b,h,t,d]
    dattn_weights = dcontext @ V.transpose(0, 1, 3, 2)  # (batch, N_HEADS, seq_len, seq_len)
    dV = attn_weights.transpose(0, 1, 3, 2) @ dcontext  # (batch, N_HEADS, seq_len, head_dim)

    # ---- 4. 反向通过 dropout ----
    if dropout_mask is not None:
        dattn_weights = dattn_weights * dropout_mask / (1.0 - DROPOUT)

    # ---- 5. 反向通过 softmax: attn_weights = softmax(scores) ----
    # softmax 的雅可比: dscore_i = softmax_i * (dout_i - sum_j dout_j * softmax_j)
    dscores = attn_weights * (dattn_weights - np.sum(dattn_weights * attn_weights, axis=-1, keepdims=True))

    # ---- 6. 反向除以 scale: scores = Q @ K^T / sqrt(head_dim) ----
    dscores = dscores / scale

    # ---- 7. 反向通过 scores = Q @ K^T ----
    # scores[b,h,s,t] = sum_d Q[b,h,s,d] * K[b,h,t,d]
    dQ = dscores @ K  # (batch, N_HEADS, seq_len, head_dim): dQ[b,h,s,d] = sum_t dscores[b,h,s,t]*K[b,h,t,d]
    dK = dscores.transpose(0, 1, 3, 2) @ Q  # (batch, N_HEADS, seq_len, head_dim): dK[b,h,t,d] = sum_s dscores[b,h,s,t]*Q[b,h,s,d]

    # ---- 8. 合并多头梯度: (batch, N_HEADS, seq_len, head_dim) → (batch, seq_len, d_model) ----
    dQ = dQ.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
    dK = dK.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
    dV = dV.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)

    # ---- 9. 反向通过线性投影 Q/K/V = x @ W ----
    x_flat = x.reshape(-1, d_model)
    dW_q = x_flat.T @ dQ.reshape(-1, d_model)  # (d_model, d_model)
    dW_k = x_flat.T @ dK.reshape(-1, d_model)
    dW_v = x_flat.T @ dV.reshape(-1, d_model)

    # dx 来自三个投影的梯度之和
    dx = dQ @ W_q.T + dK @ W_k.T + dV @ W_v.T  # (batch, seq_len, d_model)

    return dx, dW_q, dW_k, dW_v, dW_o


def feedforward_forward(x, W1, b1, W2, b2, training=True):
    """
    两层前馈网络: x → Linear(W1,b1) → ReLU → Linear(W2,b2) → output
    使用 ReLU 激活函数,简单且训练高效。
    训练时不应用 dropout(在模型外层 attention 的 dropout 已足够)。
    """
    h = x @ W1 + b1  # (batch, seq_len, D_FF)
    h_act = np.maximum(0, h)  # ReLU
    output = h_act @ W2 + b2
    cache = (x, h, h_act, W1, W2)
    return output, cache


def feedforward_backward(dout, cache):
    """
    前馈网络的反向传播。
    ReLU 正传: h_act = max(0, h)
    反传: dh_act = dout @ W2.T; dh = dh_act * (h > 0)
    """
    x, h, h_act, W1, W2 = cache

    # 反向通过第二层线性
    dW2 = h_act.reshape(-1, h_act.shape[-1]).T @ dout.reshape(-1, dout.shape[-1])
    db2 = dout.sum(axis=(0, 1))
    dh_act = dout @ W2.T

    # 反向通过 ReLU
    dh = dh_act * (h > 0).astype(np.float32)

    # 反向通过第一层线性
    dW1 = x.reshape(-1, x.shape[-1]).T @ dh.reshape(-1, dh.shape[-1])
    db1 = dh.sum(axis=(0, 1))
    dx = dh @ W1.T

    return dx, dW1, db1, dW2, db2


def transformer_block_forward(x, block_params, mask, training=True):
    """
    单个 Transformer 层的前向传播,使用 Pre-LN 结构:
    1. LayerNorm → 多头注意力 → 残差
    2. LayerNorm → 前馈网络 → 残差
    Pre-LN 比 Post-LN 训练更稳定,是 GPT-2 的标准做法。
    """
    # 子层 1: 自注意力
    h, ln1_cache = layer_norm(x, block_params['ln1_gamma'], block_params['ln1_beta'])
    attn_out, attn_cache = multi_head_attention_forward(
        h,
        block_params['W_q'], block_params['W_k'],
        block_params['W_v'], block_params['W_o'],
        mask, training=training
    )
    x = x + attn_out  # 残差连接

    # 子层 2: 前馈网络
    h2, ln2_cache = layer_norm(x, block_params['ln2_gamma'], block_params['ln2_beta'])
    ff_out, ff_cache = feedforward_forward(
        h2,
        block_params['W1'], block_params['b1'],
        block_params['W2'], block_params['b2'],
        training=training
    )
    x = x + ff_out  # 残差连接

    cache = (ln1_cache, attn_cache, ln2_cache, ff_cache)
    return x, cache


def transformer_block_backward(dout, cache):
    """
    单个 Transformer 层的反向传播,按前向的逆顺序回传梯度。
    残差连接将梯度一分为二:一条通过子层,一条直接跳过。
    """
    ln1_cache, attn_cache, ln2_cache, ff_cache = cache

    # 反向通过第二个残差连接
    dx_residual2 = dout  # 残差直接传递的梯度
    dff_out = dout  # 进入前馈网络的梯度

    # 反向通过前馈网络
    dh2, dW1, db1, dW2, db2 = feedforward_backward(dff_out, ff_cache)

    # 反向通过 LayerNorm 2
    dx_from_ln2, dln2_gamma, dln2_beta = layer_norm_backward(dh2, ln2_cache)

    # 合并梯度(残差 + LayerNorm 输出)
    dx = dx_residual2 + dx_from_ln2

    # 反向通过第一个残差连接
    dx_residual1 = dx
    dattn_out = dx

    # 反向通过多头注意力
    dh, dW_q, dW_k, dW_v, dW_o = multi_head_attention_backward(dattn_out, attn_cache)

    # 反向通过 LayerNorm 1
    dx_from_ln1, dln1_gamma, dln1_beta = layer_norm_backward(dh, ln1_cache)

    # 最终梯度
    dx = dx_residual1 + dx_from_ln1

    grads = {
        'W_q': dW_q, 'W_k': dW_k, 'W_v': dW_v, 'W_o': dW_o,
        'ln1_gamma': dln1_gamma, 'ln1_beta': dln1_beta,
        'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2,
        'ln2_gamma': dln2_gamma, 'ln2_beta': dln2_beta,
    }
    return dx, grads


def gpt_forward(x, params, training=True):
    """
    GPT 完整前向传播。
    x shape: (batch, seq_len)
    1. Token 嵌入 + 位置嵌入
    2. N 层 Transformer Block
    3. 最终 LayerNorm
    4. 通过共享权重的 LM Head 得到 logits
    返回 logits, caches, ln_f_cache
    """
    batch_size, seq_len = x.shape

    # Token 嵌入 (高级索引)
    token_emb = params['token_embedding'][x]  # (batch, seq_len, D_MODEL)

    # 位置嵌入 (取前 seq_len 个位置,broadcast 到 batch)
    pos_emb = params['position_embedding'][:seq_len]  # (seq_len, D_MODEL)

    # 嵌入相加
    h = token_emb + pos_emb  # broadcast: (batch, seq_len, D_MODEL)

    # 因果掩码
    mask = causal_mask(seq_len)

    # 逐层 Transformer
    caches = []
    for i in range(N_LAYERS):
        block_params = {
            'ln1_gamma': params[f'block_{i}_ln1_gamma'],
            'ln1_beta': params[f'block_{i}_ln1_beta'],
            'W_q': params[f'block_{i}_W_q'],
            'W_k': params[f'block_{i}_W_k'],
            'W_v': params[f'block_{i}_W_v'],
            'W_o': params[f'block_{i}_W_o'],
            'ln2_gamma': params[f'block_{i}_ln2_gamma'],
            'ln2_beta': params[f'block_{i}_ln2_beta'],
            'W1': params[f'block_{i}_W1'],
            'b1': params[f'block_{i}_b1'],
            'W2': params[f'block_{i}_W2'],
            'b2': params[f'block_{i}_b2'],
        }
        h, cache = transformer_block_forward(h, block_params, mask, training=training)
        caches.append(cache)

    # 最终 LayerNorm
    h, ln_f_cache = layer_norm(h, params['ln_f_gamma'], params['ln_f_beta'])

    # LM Head (与 token_embedding 共享权重)
    logits = h @ params['token_embedding'].T  # (batch, seq_len, VOCAB_SIZE)

    # 返回 h_forward(LayerNorm 输出)供反向传播使用
    return logits, caches, ln_f_cache, h


# ===========================================================================
# 模块 3: 损失函数
# ===========================================================================

def cross_entropy_loss(logits, targets):
    """
    交叉熵损失的计算与 cache 存储。
    logits shape: (batch, seq_len, VOCAB_SIZE)
    targets shape: (batch, seq_len)
    使用 log-sum-exp 技巧保证数值稳定性。
    返回平均 loss 和 cache(用于反向传播)。
    """
    batch, seq_len, vocab_size = logits.shape
    N = batch * seq_len

    # Reshape 为 2D
    logits_2d = logits.reshape(N, vocab_size)
    targets_1d = targets.reshape(N)

    # 数值稳定: 减去最大值
    logits_max = logits_2d.max(axis=-1, keepdims=True)
    logits_stable = logits_2d - logits_max

    # log-sum-exp
    log_sum_exp = np.log(np.exp(logits_stable).sum(axis=-1))

    # 采集每个样本对应目标 token 的 logit
    target_logits = logits_stable[np.arange(N), targets_1d]

    # 交叉熵 = -log p(target) = -target_logit + log(sum(exp(logits)))
    losses = -target_logits + log_sum_exp
    loss = losses.mean()

    # 存储 cache 用于反向传播(保留 batch 和 seq_len 信息)
    cache = (logits_stable, targets_1d, N, vocab_size, batch, seq_len)
    return loss, cache


def cross_entropy_backward(loss_cache):
    """
    交叉熵损失对 logits 的梯度。
    d_logits[i, target_i] = softmax[i, target_i] - 1
    然后对所有样本取平均 (除以 N)。
    返回 dlogits shape: (batch, seq_len, VOCAB_SIZE)
    """
    logits_stable, targets_1d, N, vocab_size, batch, seq_len = loss_cache

    # softmax 概率
    logits_max = logits_stable.max(axis=-1, keepdims=True)
    e_x = np.exp(logits_stable - logits_max)
    probs = e_x / e_x.sum(axis=-1, keepdims=True)

    # 对目标位置减 1 (梯度 = p - y_onehot)
    probs[np.arange(N), targets_1d] -= 1.0

    # 平均梯度 (因为 loss 是 mean),并 reshape 回 3D
    dlogits = (probs / N).reshape(batch, seq_len, vocab_size)

    return dlogits


# ===========================================================================
# 模块 4: 优化器
# ===========================================================================

class AdamOptimizer:
    """
    Adam 优化器,支持 warmup 和 cosine 学习率衰减。
    对每个参数维护一阶矩(m)和二阶矩(v)的指数移动平均。
    偏差校正确保初期更新不偏向零。
    """

    def __init__(self, params):
        self.m = {key: np.zeros_like(val) for key, val in params.items()}
        self.v = {key: np.zeros_like(val) for key, val in params.items()}
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def get_lr(self, step, warmup_steps, base_lr, total_steps):
        """
        学习率调度:
        - warmup 阶段: 线性增加
        - 之后: cosine 衰减到接近 0
        """
        if step < warmup_steps:
            return base_lr * step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return base_lr * 0.5 * (1 + np.cos(np.pi * progress))

    def step(self, params, grads, lr, t):
        """
        执行一步 Adam 更新。
        t: 当前步数(从 1 开始),用于偏差校正。
        """
        for key in params:
            if key not in grads:
                continue
            # 更新一阶和二阶矩估计
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2

            # 偏差校正
            m_hat = self.m[key] / (1 - self.beta1 ** t)
            v_hat = self.v[key] / (1 - self.beta2 ** t)

            # 更新参数
            params[key] -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


def clip_gradients(grads, max_norm):
    """
    按全局范数裁剪梯度,防止梯度爆炸。
    如果梯度的总范数超过 max_norm,则按比例缩放。
    返回裁剪后的 grads 和(可选)原始范数。
    """
    # 计算所有梯度的全局 L2 范数
    total_norm = 0.0
    for g in grads.values():
        total_norm += np.sum(g ** 2)
    total_norm = np.sqrt(total_norm)

    # 如果范数超过阈值,缩放所有梯度
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for key in grads:
            grads[key] *= scale

    return grads


# ===========================================================================
# 模块 5: 训练循环
# ===========================================================================

def save_model(params, path):
    """保存模型参数到 .npz 文件"""
    np.savez(path, **params)
    print(f"[保存] 模型已保存到 {path}")


def load_model(path):
    """从 .npz 文件加载模型参数"""
    data = np.load(path)
    params = {key: data[key] for key in data.files}
    print(f"[加载] 从 {path} 加载了 {len(params)} 个参数张量")
    return params


def train_step(params, inputs_batch, targets_batch, optimizer, step, total_steps):
    """
    单步训练: 前向 → 损失 → 反向 → 梯度裁剪 → 参数更新。
    这是整个训练的核心,需要正确处理梯度流。
    """
    grads = {}

    # ============== 前向传播 ==============
    logits, caches, ln_f_cache, h_forward = gpt_forward(inputs_batch, params, training=True)

    # 计算损失
    loss, loss_cache = cross_entropy_loss(logits, targets_batch)

    # ============== 反向传播 ==============
    # 1. 反向通过交叉熵损失
    dlogits = cross_entropy_backward(loss_cache)  # (batch, seq_len, VOCAB_SIZE)
    batch_size, seq_len, vocab_size = dlogits.shape

    # h_forward 是 LayerNorm 输出(即 LM Head 的输入)

    # 2. 反向通过共享权重的 LM Head
    # logits = h_forward @ token_embedding.T
    # dh = dlogits @ token_embedding (W = embedding.T, 所以 dx = dout @ W.T... 不对)
    # 正确: logits = h @ W, W = embedding.T (D_MODEL, VOCAB_SIZE)
    # dh = dlogits @ W.T... 不对,应该是 dh = dlogits @ W... 再检查
    # logits[b,s,v] = sum_d h_forward[b,s,d] * token_embedding[v,d]
    # 所以 dh_forward[b,s,d] = sum_v dlogits[b,s,v] * token_embedding[v,d]
    # 即 dh_forward = dlogits @ token_embedding
    # token_embedding 的梯度: dW[v,d] = sum_{b,s} dlogits[b,s,v] * h_forward[b,s,d]
    # → d_token_embedding = dlogits_flat.T @ h_forward_flat (VOCAB_SIZE, D_MODEL)
    h_flat = h_forward.reshape(-1, D_MODEL)
    dlogits_flat = dlogits.reshape(-1, vocab_size)
    dh_from_head = dlogits @ params['token_embedding']  # (batch, seq_len, D_MODEL)
    # 共享权重的 LM Head 梯度: grad_token_embedding += dlogits.T @ h_forward
    grads['token_embedding'] = dlogits_flat.T @ h_flat  # (VOCAB_SIZE, D_MODEL)

    # 3. 反向通过最终的 LayerNorm
    dh, d_ln_f_gamma, d_ln_f_beta = layer_norm_backward(dh_from_head, ln_f_cache)
    grads['ln_f_gamma'] = d_ln_f_gamma
    grads['ln_f_beta'] = d_ln_f_beta

    # 4. 反向通过各 Transformer 层(逆序)
    for i in reversed(range(N_LAYERS)):
        dh, block_grads = transformer_block_backward(dh, caches[i])
        prefix = f'block_{i}'
        for key, val in block_grads.items():
            grads[f'{prefix}_{key}'] = val

    # 5. dh 现在是反向到初始嵌入 (token_emb + pos_emb) 的梯度
    # Token embedding 梯度:从高级索引反向,将梯度 scatter 到对应位置
    d_token_embedding = np.zeros_like(params['token_embedding'])
    np.add.at(d_token_embedding, inputs_batch, dh)  # 正确处理重复索引
    grads['token_embedding'] += d_token_embedding

    # 位置嵌入梯度:对所有 batch 求和(因为位置嵌入在所有 batch 中共享)
    d_pos_emb = dh.sum(axis=0)  # (seq_len, D_MODEL)
    grads['position_embedding'] = np.zeros_like(params['position_embedding'])
    grads['position_embedding'][:seq_len] = d_pos_emb

    # ============== 梯度裁剪 ==============
    grads = clip_gradients(grads, CLIP_GRAD)

    # ============== 学习率调度 ==============
    lr = optimizer.get_lr(step, WARMUP_STEPS, LEARNING_RATE, total_steps)

    # ============== 参数更新 ==============
    optimizer.step(params, grads, lr, step + 1)

    return loss


def train(params, inputs, targets, char2idx, idx2char):
    """
    完整训练循环。
    每个 epoch 遍历所有 batch,共训练 NUM_EPOCHS 轮。
    """
    optimizer = AdamOptimizer(params)
    num_samples = inputs.shape[0]
    steps_per_epoch = num_samples // BATCH_SIZE
    total_steps = steps_per_epoch * NUM_EPOCHS

    print(f"\n{'='*60}")
    print(f"开始训练: {NUM_EPOCHS} epochs, 每 epoch {steps_per_epoch} 步, 共 {total_steps} 步")
    print(f"{'='*60}\n")

    start_time = time.time()
    best_loss = float('inf')

    for step in range(total_steps):
        # 采样一个 batch
        batch_input, batch_target = get_batch(inputs, targets, BATCH_SIZE)

        # 训练一步
        loss = train_step(params, batch_input, batch_target, optimizer, step, total_steps)

        # 定期打印训练信息
        if step % PRINT_EVERY == 0:
            elapsed = time.time() - start_time
            lr = optimizer.get_lr(step, WARMUP_STEPS, LEARNING_RATE, total_steps)
            print(f"  Step {step:6d}/{total_steps} | loss: {loss:.4f} | lr: {lr:.6f} | time: {elapsed:.1f}s")

        # 定期保存模型
        if step % SAVE_EVERY == 0 and step > 0:
            save_model(params, MODEL_SAVE_PATH)

        # 记录最佳 loss
        if loss < best_loss:
            best_loss = loss

    total_time = time.time() - start_time
    print(f"\n训练完成! 总时间: {total_time:.1f}s, 最佳 loss: {best_loss:.4f}")

    # 保存最终模型
    save_model(params, MODEL_SAVE_PATH)
    return params


# ===========================================================================
# 模块 6: 文本生成
# ===========================================================================

# 简繁映射：训练语料为全唐诗（繁体），prompt 可能含简体字
SIMP_TO_TRAD = {
    '东': '東', '红': '紅', '觉': '覺', '晓': '曉', '见': '見',
    '风': '風', '云': '雲', '辞': '辭', '乐': '樂', '长': '長',
    '时': '時', '诗': '詩', '书': '書', '词': '詞', '语': '語',
    '车': '車', '马': '馬', '门': '門', '国': '國', '学': '學',
}


def generate(prompt, params, char2idx, idx2char, max_new_tokens=100, temperature=0.8, top_k=50):
    """
    基于 prompt 生成文本。
    支持 temperature(控制多样性) 和 top_k(限制采样范围) 采样。
    推理时不应用 dropout(training=False)。
    """
    # prompt 简体转繁体，以匹配训练语料
    prompt_trad = ''.join(SIMP_TO_TRAD.get(c, c) for c in prompt)
    tokens = [char2idx.get(c, 1) for c in prompt_trad]  # 转为 token id

    for _ in range(max_new_tokens):
        # 取最后 MAX_SEQ_LEN 个 token 作为输入
        input_tokens = tokens[-MAX_SEQ_LEN:]
        x = np.array([input_tokens], dtype=np.int32)  # (1, seq_len)

        # 前向传播
        logits, _, _, _ = gpt_forward(x, params, training=False)

        # 取最后一个位置的 logits
        next_logits = logits[0, -1, :]  # (VOCAB_SIZE,)

        # Temperature 缩放(越高越多样,越低越确定)
        next_logits = next_logits / temperature

        # Top-k 采样: 只从概率最高的 k 个中选择
        # 获取 top-k 的阈值
        top_k_vals = np.partition(next_logits, -top_k)[-top_k]
        next_logits[next_logits < top_k_vals] = -np.inf

        # Softmax 得到采样概率
        probs = softmax(next_logits)

        # 采样下一个 token
        next_token = np.random.choice(len(probs), p=probs)

        tokens.append(next_token)

        # 遇到 <EOS> 停止生成
        if next_token == char2idx.get('<EOS>', 3):
            break

    # 将 token 转回文本(idx2char 是列表,直接索引访问)
    result = ''.join([idx2char[t] if t < len(idx2char) else '' for t in tokens])
    return result


# ===========================================================================
# 模块 7: 主函数
# ===========================================================================

def main():
    """主入口函数"""
    print("=" * 60)
    print("  中文 GPT 模型 —— 纯 NumPy 从零实现")
    print(f"  模型: {N_LAYERS} 层, {D_MODEL} 维, {N_HEADS} 头, ~1M 参数")
    print("=" * 60)
    print()

    # 1. 下载/准备语料
    print("[步骤 1] 准备语料...")
    text = download_corpus()

    # 2. 构建词汇表
    print("\n[步骤 2] 构建词汇表...")
    char2idx, idx2char = build_vocab(text, VOCAB_SIZE)

    # 3. 准备训练数据
    print("\n[步骤 3] 准备训练数据...")
    inputs, targets = prepare_data(text, char2idx, MAX_SEQ_LEN)
    print(f"        输入形状: {inputs.shape}, 目标形状: {targets.shape}")

    # 4. 初始化模型
    print("\n[步骤 4] 初始化模型...")
    params = init_model()

    # 5. 如果存在已保存的模型则加载
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"\n[步骤 5] 发现已保存模型,加载中...")
        try:
            saved_params = load_model(MODEL_SAVE_PATH)
            # 只加载匹配的参数
            for key in params:
                if key in saved_params:
                    params[key] = saved_params[key]
            print("        模型加载成功,继续训练")
        except Exception as e:
            print(f"        加载失败: {e},使用初始参数训练")

    # 6. 训练
    print("\n[步骤 6] 开始训练...")
    params = train(params, inputs, targets, char2idx, idx2char)

    # 7. 文本生成演示
    print("\n" + "=" * 60)
    print("  文本生成演示")
    print("=" * 60)

    prompts = [
        "春眠不觉晓",
        "人生若只如初见",
        "人工智能",
        "大江东去",
        "红色",
    ]

    for prompt in prompts:
        print(f"\n上文人: {prompt}")
        result = generate(prompt, params, char2idx, idx2char, max_new_tokens=50, temperature=0.5, top_k=20)
        print(f"生成: {result}")

    print("\n" + "=" * 60)
    print("  全部完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 允许通过环境变量覆盖 epochs（供训练脚本使用）
    env_epochs = os.environ.get("GPT_EPOCHS")
    if env_epochs:
        import sys
        this_mod = sys.modules[__name__]
        this_mod.NUM_EPOCHS = int(env_epochs)
    main()
