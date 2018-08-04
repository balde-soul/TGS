import tensorflow as tf
import Putil.tf.util as tfu
from colorama import Fore
import Putil.loger as plog
import model_base as mb
import Putil.tf.model_helper as tfmh
import os

logger_model = plog.PutilLogConfig('TGS/Model').logger()
logger_model.setLevel(plog.DEBUG)

logger_model_class = logger_model.getChild("ModelClass")
logger_model_class.setLevel(plog.DEBUG)


class Model:
    def __init__(self, param):
        self._train_result_reflect = ['loss', 'acc']
        self._val_result_reflect = ['loss', 'acc']
        # {'param_type',
        self._param = param
        self._param_UNet = self._param['U-Net']
        self._param_opt = self._param['opt']
        self._param_loss = self._param_UNet['Loss']
        self._param_training = self._param['Training']
        self._param_training["SummaryPath"], self._param_training['WeightPath'] = \
            tfmh.SavePathProcess(self._param_training['SavePath'])
        # {'image', 'GT'
        self._placehold = dict()
        self._sess = None
        self._loss = None
        self._acc = None
        self._opt = None
        self._step = None
        self._train_op = None

        self._batch = None
        self._graph_saved = False
        self._train_summary_op = None
        self._val_summary_op = None
        self._pro = None
        self._writer = None
        pass

    def __task_placeholder(self):

        self._placehold['image'] = tf.placeholder(dtype=tfu.tf_type(self._param['param_dtype']).Type,
                                                  shape=[None, None, None, 1], name='image')

        self._placehold['GT'] = tf.placeholder(dtype=tfu.tf_type(self._param['param_dtype']).Type,
                                               shape=[None, None, None, self._param['class_amount']],
                                               name='GT')

        self._placehold['training'] = tf.placeholder(dtype=tf.bool, name='training')

        pass

    def __make_train_feed(self, data):
        feed = dict()
        for i in data.keys():
            feed[self._placehold[i]] = data[i]
            pass
        feed[self._placehold['training']] = True
        return feed
        pass

    def __make_val_feed(self, data):
        feed = dict()
        for i in data.keys():
            feed[self._placehold[i]] = data[i]
            pass
        feed[self._placehold['training']] = False
        return feed
        pass

    # : display the placeholder information
    def __display_placeholder(self):
        info = '-->placehold:\n'
        for i in self._placehold.keys():
            info += i + ', '
        logger_model_class.info(
            Fore.GREEN + info
        )
        pass

    # : build dense_UNet with loss
    def __build_Dense_UNet_with_loss(self):
        output_map = mb.DenseUNetPro(
            self._placehold['image'],
            training=self._placehold['training'],
            class_amount=self._param['class_amount'],
            param_dtype=tfu.tf_type(self._param['param_dtype']).Type,
            regularizer_weight=self._param['regularizer_weight'],
            DenseUNetConfig=self._param_UNet
        )
        self._pro = output_map
        self._acc = mb.fcn_acc(
            output_map,
            self._placehold['GT'],
        )
        cw = self._param_loss.get('class_weights', None)
        self._loss = mb.fcn_loss(
            output_map,
            self._placehold['GT'],
            cost_name=self._param_loss['cost_name'],
            param_dtype=self._param['param_dtype'],
            class_weights=cw
        )
        pass

    def __build_with_loss(self):
        if self._param['BaseModel'] == 'DenseNet':
            self.__build_Dense_UNet_with_loss()
            pass
        else:
            print(Fore.RED + 'base model is not supported')
        # after build display placeholder
        self.__display_placeholder()
        pass

    def __build_Adam_opt(self):
        param = self._param_opt['opt_param']
        self._opt = tf.train.AdamOptimizer(
            param['learning_lr'],
            param['beta1'],
            param['beta2'],
            param['epsilon']
        )
        pass

    def __build_opt(self):
        if self._param['opt']['opt'] == 'Adam':
            self.__build_Adam_opt()
        else:
            raise ValueError('opt: {0} is not supported now'.format(self._param['opt']['opt']))
        pass

    # :every cv run this to init the model
    def re_init(self):
        self._batch = 0
        # : reset the model
        tf.reset_default_graph()
        # generate placeholder
        self.__task_placeholder()
        logger_model_class.info('placeholder keys: {0}'.format(
            self._placehold.keys()
        ))
        # : rebuild the model
        self.__build_with_loss()
        # : init the session and step
        self._sess = tf.Session()
        self._step = tf.Variable(0, trainable=False)
        # : generate the opt
        self.__build_opt()
        # : apply moving average and mv_op
        _ema = tf.train.ExponentialMovingAverage(self._param['moving_decay'], self._step)
        # : generator train_op dependent on mv_op
        with tf.control_dependencies([_ema.apply(tf.trainable_variables())]):
            self._train_op = self._opt.minimize(self._loss, self._step)
            pass
        self._sess.run(tf.global_variables_initializer())
        if self._graph_saved is False:
            tfmh.save_graph_and_pause(self._param_training['SummaryPath'])
            self._graph_saved = True
            pass
        tf.GraphKeys.TrainSummary = 'TrainSummary'
        tf.GraphKeys.ValSummary = 'ValSummary'
        tf.add_to_collection(tf.GraphKeys.TrainSummary, tf.summary.histogram("train_pro", self._pro))
        tf.add_to_collection(tf.GraphKeys.ValSummary, tf.summary.histogram("val_pro", self._pro))
        tf.add_to_collection(tf.GraphKeys.TrainSummary, tf.summary.scalar("train_loss", self._loss))
        tf.add_to_collection(tf.GraphKeys.ValSummary, tf.summary.scalar("val_loss", self._loss))
        tf.add_to_collection(tf.GraphKeys.TrainSummary, tf.summary.scalar("train_acc", self._acc))
        tf.add_to_collection(tf.GraphKeys.ValSummary, tf.summary.scalar("val_acc", self._acc))
        self._train_summary_op = tf.summary.merge_all(tf.GraphKeys.TrainSummary)
        self._val_summary_op = tf.summary.merge_all(tf.GraphKeys.ValSummary)
        i = 0
        while True:
            path = os.path.join(self._param_training['SummaryPath'], str(i))
            if os.path.exists(path):
                i += 1
                continue
            else:
                self._writer = tf.summary.FileWriter(path)
                break
                pass
            pass
        pass

    # : run one time
    def TrainCV(self, data):
        # : feed the placeholder
        feed = self.__make_train_feed(data)
        # : tun the loss and train and ***
        _loss, _acc, summary, _ = self._sess.run(
            [self._loss, self._acc, self._train_summary_op, self._train_op],
            feed_dict=feed)
        self._batch += 1
        # : check summary or not
        if self._batch % self._param_training['SummaryBatch'] == 0:
            self._writer.add_summary(summary, self._batch)
            pass
        if self._batch % self._param_training['DisplayBatch'] == 0:
            print(Fore.GREEN + 'Batch: {0}, Loss: {1}, acc: {2}'.format(self._batch, _loss, _acc))
            pass
        # : return the result want to estimate: TrainResultReflect
        return {'loss': _loss, 'acc': _acc}
        pass

    # :val one time
    def Val(self, data):
        # : feed the placeholder
        feed = self.__make_val_feed(data)
        # : tun the loss and train and ***
        _loss, _acc, summary = self._sess.run([self._loss, self._acc, self._val_summary_op], feed_dict=feed)
        # : return the result want to estimate: ValResultReflect
        return {'loss': _loss, 'acc': _acc}
        pass

    @property
    def TrainResultReflect(self):
        return self._train_result_reflect
        pass

    @property
    def ValResultReflect(self):
        return self._val_result_reflect
        pass
    pass


if __name__ == '__main__':
    pass
