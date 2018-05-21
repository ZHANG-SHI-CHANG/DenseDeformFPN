import tensorflow as tf
from tensorflow.python.framework import graph_util

import numpy as np
import cv2

#from DenseDeformConvFPN_64_48 import DenseFPN,Detection_or_Classifier
#from DenseDeformConvFPN_121_24 import DenseFPN,Detection_or_Classifier
#from DenseDeformConvFPN_121_12 import DenseFPN,Detection_or_Classifier
from DenseFPN_64_48 import DenseFPN,Detection_or_Classifier
#from DenseFPN_121_24 import DenseFPN,Detection_or_Classifier
#from DenseFPN_121_12 import DenseFPN,Detection_or_Classifier

from detection_dataloader import create_training_instances,BatchGenerator
from classifier_dataloader import DataLoader

import os
import glob

from functools import reduce
from operator import mul
def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params

num_epochs = 10000
max_to_keep = 2
save_model_every = 1
test_every = 1

is_train = True

##############################################
anchors = [18,27, 28,75, 49,132, 55,43, 65,227, 84,86, 108,162, 109,288, 162,329, 174,103, 190,212, 245,348, 321,150, 343,256, 372,379]
max_box_per_image = 60
min_input_size = 224
max_input_size = 480
batch_size = 2
ignore_thresh = 0.6

first_train = False
##############################################

def normalize(image):
    return image/255.

def main():
    # Reset the default Tensorflow graph
    tf.reset_default_graph()

    # Tensorflow specific configuration
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = False
    sess = tf.Session(config=config)

    # Train class
    trainer = Train(sess)

    if is_train:
        try:
            print("Training...")
            trainer.train()
            print("Training Finished\n\n")
        except KeyboardInterrupt:
            trainer.save_model()
    else:
        print("Testing...")
        trainer.test()
        print("Testing Finished\n\n")

class Train:
    """Trainer class for the CNN.
    It's also responsible for loading/saving the model checkpoints from/to experiments/experiment_name/checkpoint_dir"""

    def __init__(self, sess):
        self.sess = sess
        self.dataset_root = 'F:\\Learning\\tensorflow\\detect\\Dataset\\'
        
        if Detection_or_Classifier=='classifier':
            self.train_data = DataLoader(root=self.dataset_root+'SmallNORB\\trainImages',batch=batch_size)
            self.test_data = DataLoader(root=self.dataset_root+'SmallNORB\\testImages',batch=batch_size)
            
            labels = ['1','2','3','4','5']
            
            print("Building the model...")
            self.model = DenseFPN(num_classes=len(labels),
                                  num_anchors=5,
                                  batch_size = batch_size,
                                  max_box_per_image = max_box_per_image,
                                  max_grid=[max_input_size,max_input_size],
                                  )
            print("Model is built successfully\n\n")
            
            
        elif Detection_or_Classifier=='detection':
            train_ints, valid_ints, labels = create_training_instances(
            self.dataset_root+'VOC2012\\Annotations\\',
            self.dataset_root+'VOC2012\\JPEGImages\\',
            'data.pkl',
            '','','',
            ['person','head','hand','foot','aeroplane','tvmonitor','train','boat','dog','chair',
             'bird','bicycle','bottle','sheep','diningtable','horse','motorbike','sofa','cow',
             'car','cat','bus','pottedplant']
            )
            self.train_data = BatchGenerator(
                                            instances           = train_ints, 
                                            anchors             = anchors,   
                                            labels              = labels,        
                                            downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
                                            max_box_per_image   = max_box_per_image,
                                            batch_size          = batch_size,
                                            min_net_size        = min_input_size,
                                            max_net_size        = max_input_size,   
                                            shuffle             = True, 
                                            jitter              = 0.3, 
                                            norm                = normalize
                                            )
            self.test_data = BatchGenerator(
                                            instances           = valid_ints, 
                                            anchors             = anchors,   
                                            labels              = labels,        
                                            downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
                                            max_box_per_image   = max_box_per_image,
                                            batch_size          = batch_size,
                                            min_net_size        = min_input_size,
                                            max_net_size        = max_input_size,   
                                            shuffle             = True, 
                                            jitter              = 0.0, 
                                            norm                = normalize
                                            )
            
            print("Building the model...")
            self.model = DenseFPN(num_classes=len(labels),
                                  num_anchors=5,
                                  batch_size = batch_size,
                                  max_box_per_image = max_box_per_image,
                                  max_grid=[max_input_size,max_input_size],
                                  )
            print("Model is built successfully\n\n")
        
        #tf.profiler.profile(tf.get_default_graph(),options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter(), cmd='scope')
        
        num_params = get_num_params()
        print('all params:{}'.format(num_params))
        
        var = tf.global_variables()
        var_list = [val for val in var]
        if Detection_or_Classifier=='detection' and first_train:
            var_list = [val for val in var if ('zsc_preprocessing' in val.name) or ('zsc_feature' in val.name)]
        
        self.saver = tf.train.Saver(var_list=var_list,max_to_keep=max_to_keep,
                                    keep_checkpoint_every_n_hours=10)
        
        self.save_checkpoints_path = os.path.join(os.getcwd(),'checkpoints',Detection_or_Classifier)
        if not os.path.exists(self.save_checkpoints_path):
            os.makedirs(self.save_checkpoints_path)

        # Initializing the model
        self.init = None
        self.__init_model()

        # Loading the model checkpoint if exists
        self.__load_model()
        
        summary_dir = os.path.join(os.getcwd(),'logs',Detection_or_Classifier)
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        summary_dir_train = os.path.join(summary_dir,'train')
        if not os.path.exists(summary_dir_train):
            os.makedirs(summary_dir_train)
        summary_dir_test = os.path.join(summary_dir,'test')
        if not os.path.exists(summary_dir_test):
            os.makedirs(summary_dir_test)
        self.train_writer = tf.summary.FileWriter(summary_dir_train,sess.graph)
        self.test_writer = tf.summary.FileWriter(summary_dir_test)

    ############################################################################################################
    # Model related methods
    def __init_model(self):
        print("Initializing the model...")
        self.init = tf.group(tf.global_variables_initializer())
        self.sess.run(self.init)
        print("Model initialized\n\n")

    def save_model(self):
        var = tf.global_variables()
        var_list = [val for val in var]
        self.saver = tf.train.Saver(var_list=var_list,max_to_keep=max_to_keep,
                                    keep_checkpoint_every_n_hours=10)
        
        print("Saving a checkpoint")
        self.saver.save(self.sess, self.save_checkpoints_path+'\\'+Detection_or_Classifier, self.model.global_step_tensor)
        print("Checkpoint Saved\n\n")
        
        print('Saving a pb')
        if Detection_or_Classifier=='classifier':
            output_graph_def = graph_util.convert_variables_to_constants(self.sess, self.sess.graph.as_graph_def(), ['output/zsc_output'])
            #tflite_model = tf.contrib.lite.toco_convert(output_graph_def, [self.model.input_image], [self.model.y_out_softmax])
            #open(Detection_or_Classifier+".tflite", "wb").write(tflite_model)
        elif Detection_or_Classifier=='detection':
            output_graph_def = graph_util.convert_variables_to_constants(self.sess, self.sess.graph.as_graph_def(), ['zsc_output'])
        tf.train.write_graph(output_graph_def, self.save_checkpoints_path, Detection_or_Classifier+'.pb', as_text=False)
        print('pb saved\n\n')
        
    def __load_model(self):
        if Detection_or_Classifier=='detection' and first_train:
            latest_checkpoint = tf.train.latest_checkpoint(os.path.join(os.getcwd(),'checkpoints','classifier'))
            if latest_checkpoint:
                print("loading classifier checkpoint {} ...\n".format(latest_checkpoint))
                self.saver.restore(self.sess, latest_checkpoint)
                print("classifier model success loaded\n\n")
            else:
                print('loading classifier model failure!!')
        else:
            latest_checkpoint = tf.train.latest_checkpoint(self.save_checkpoints_path)
            if latest_checkpoint:
                print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
                self.saver.restore(self.sess, latest_checkpoint)
                print("Checkpoint loaded\n\n")
            else:
                print("First time to train!\n\n")
    ############################################################################################################
    # Train and Test methods
    def train(self):
        for cur_epoch in range(self.model.global_epoch_tensor.eval(self.sess) + 1, num_epochs + 1, 1):

            batch = 0
            
            loss_list = []
            
            if Detection_or_Classifier=='classifier':
                acc_list = []
                for X_batch, y_batch in self.train_data.next():
                    print('Training epoch:{},batch:{}\n'.format(cur_epoch,batch))
                    
                    cur_step = self.model.global_step_tensor.eval(self.sess)
                    
                    feed_dict = {self.model.input_image: X_batch,
                                 self.model.y: y_batch,
                                 self.model.is_training: 1.0
                                 }
                                 
                    _, loss, acc, summaries_merged = self.sess.run(
                    [self.model.train_op, self.model.all_loss, self.model.accuracy, self.model.summaries_merged],
                    feed_dict=feed_dict)
                    
                    print('loss:' + str(loss)+'|'+'accuracy:'+str(acc))
                    
                    loss_list += [loss]
                    acc_list += [acc]
                    
                    self.model.global_step_assign_op.eval(session=self.sess,
                                                          feed_dict={self.model.global_step_input: cur_step + 1})

                    self.train_writer.add_summary(summaries_merged,cur_step)

                    if batch > self.train_data.__len__():
                        batch = 0
                    
                        avg_loss = np.mean(loss_list).astype(np.float32)
                        avg_accuracy = np.mean(acc_list).astype(np.float32)
                        
                        self.model.global_epoch_assign_op.eval(session=self.sess,
                                                               feed_dict={self.model.global_epoch_input: cur_epoch + 1})

                        print("\nEpoch-" + str(cur_epoch) + '|' + 'avg loss:' + str(avg_loss)+'|'+'avg accuracy:'+str(avg_accuracy)+'\n')
                        break
                    
                    if batch==0 and cur_epoch%99==0:
                        #opts = tf.profiler.ProfileOptionBuilder.float_operation()    
                        #flops = tf.profiler.profile(tf.get_default_graph(), run_meta=tf.RunMetadata(), cmd='op', options=opts)
                        #if flops is not None:
                        #    print('flops:{}'.format(flops.total_float_ops))
                    
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        
                        _,summaries_merged = self.sess.run([self.model.train_op, self.model.summaries_merged],
                                                       feed_dict=feed_dict,
                                                       options=run_options,
                                                       run_metadata=run_metadata)
                        
                        self.train_writer.add_run_metadata(run_metadata, 'epoch{}batch{}'.format(cur_epoch,cur_step))
                        self.train_writer.add_summary(summaries_merged, cur_step)

                    batch += 1
                
                if cur_epoch % save_model_every == 0 and cur_epoch != 0:
                    self.save_model()
                    
            elif Detection_or_Classifier=='detection':
                for input_list,dummy_yolo in self.train_data.next():
                    print('Training epoch:{},batch:{}\n'.format(cur_epoch,batch))
                    
                    cur_step = self.model.global_step_tensor.eval(self.sess)

                    x_batch, anchors_batch,t_batch, yolo_1, yolo_2, yolo_3 = input_list
                    
                    feed_dict = {self.model.input_image:x_batch,
                                 self.model.anchors:anchors_batch,
                                 self.model.is_training:1.0,
                                 self.model.true_boxes:t_batch,
                                 self.model.true_yolo_1:yolo_1,
                                 self.model.true_yolo_2:yolo_2,
                                 self.model.true_yolo_3:yolo_3
                                 }
                    
                    _, loss, summaries_merged = self.sess.run(
                        [self.model.train_op, self.model.all_loss, self.model.summaries_merged],
                        feed_dict=feed_dict)
                    
                    loss_list += [loss]

                    self.model.global_step_assign_op.eval(session=self.sess,
                                                          feed_dict={self.model.global_step_input: cur_step + 1})

                    self.train_writer.add_summary(summaries_merged,cur_step)

                    if batch > self.train_data.__len__():
                        batch = 0
                    
                        avg_loss = np.mean(loss_list).astype(np.float32)
                        
                        self.model.global_epoch_assign_op.eval(session=self.sess,
                                                               feed_dict={self.model.global_epoch_input: cur_epoch + 1})

                        print("Epoch-" + str(cur_epoch) + " | " + "loss: " + str(avg_loss) )
                        break
                    
                    if batch==0 and cur_epoch%99==0:
                        #opts = tf.profiler.ProfileOptionBuilder.float_operation()    
                        #flops = tf.profiler.profile(tf.get_default_graph(), run_meta=tf.RunMetadata(), cmd='op', options=opts)
                        #if flops is not None:
                        #    print('flops:{}'.format(flops.total_float_ops))
                    
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        
                        _,summaries_merged = self.sess.run([self.model.train_op, self.model.summaries_merged],
                                                       feed_dict=feed_dict,
                                                       options=run_options,
                                                       run_metadata=run_metadata)
                        
                        self.train_writer.add_run_metadata(run_metadata, 'epoch{}batch{}'.format(cur_epoch,cur_step))
                        self.train_writer.add_summary(summaries_merged, cur_step)

                    batch += 1
                
                if cur_epoch % save_model_every == 0 and cur_epoch != 0:
                    self.save_model()
                
                if cur_epoch % test_every == 0:
                    print('start test')
                    self.test()
                    print('end test')
    def test(self):
        if Detection_or_Classifier=='classifier':
            labels = ['1','2','3','4','5']
            
            
            
        elif Detection_or_Classifier=='detection':
            labels = ['person','head','hand','foot','aeroplane','tvmonitor','train','boat','dog','chair',
                      'bird','bicycle','bottle','sheep','diningtable','horse','motorbike','sofa','cow',
                      'car','cat','bus','pottedplant']
            
            if not os.path.exists(os.path.join(os.getcwd(),'test_results',Detection_or_Classifier)):
                os.makedirs(os.path.join(os.getcwd(),'test_results',Detection_or_Classifier))
        
            for image_path in glob.glob(os.path.join(os.getcwd(),'test_images',Detection_or_Classifier,'*.jpg')):
                image_name = image_path.split('\\')[-1]
                print('processing image {}'.format(image_name))

                image = cv2.imread(image_path)
                image_h,image_w,_ = image.shape
                _image = cv2.resize(image,(32*9,32*9))[np.newaxis,:,:,::-1]
                infos = self.sess.run(self.model.infos,
                                      feed_dict={self.model.input_image:_image,
                                                 self.model.is_training:0.0,
                                                 self.model.original_wh:[[image_w,image_h]]
                                                 }
                                      )
                
                image = self.draw_boxes(image, infos.tolist(), labels)
                cv2.imwrite(os.path.join(os.getcwd(),'test_results',image_name),image)
    def draw_boxes(self,image, boxes, labels, obj_thresh=0.9):
        def _constrain(min_v, max_v, value):
            if value < min_v: return min_v
            if value > max_v: return max_v
            return value
        
        for box in boxes:
            label_str = ''
            label = -1
            
            for i in range(len(labels)):
                if box[5:][i] > obj_thresh:
                    label_str += labels[i]
                    label = i
                    print(labels[i] + ': ' + str(box[5:][i]*100) + '%')
                    
            if label >= 0:
                cv2.rectangle(image, (int(_constrain(1,image.shape[1]-2,box[0])),int(_constrain(1,image.shape[0]-2,box[1]))), 
                                     (int(_constrain(1,image.shape[1]-2,box[2])),int(_constrain(1,image.shape[0]-2,box[3]))), (0,255,0), 3)
                cv2.putText(image, 
                            label_str + ' ' + str(max(box[5:])), 
                            (int(box[0]), int(box[1]) - 13), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1e-3 * image.shape[0], 
                            (0,255,0), 2)
        return image

if __name__=='__main__':
    main()