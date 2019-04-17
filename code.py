import tensorflow as tf
import sys
import os
import cv2

def get_labels():
    with open('retrained_labels.txt','r') as fin:
        labels = [line.strip('\n') for line in fin]
        return labels


with tf.gfile.FastGFile('retrained_graph.pb','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def,name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    hits = 0
    misses = 0
    errors = 0
    labels = get_labels()
    for d in labels:
        for (dirpath,dirnames,filenames) in os.walk('captures/%s' %d):
            for filename in filenames:
                image = 'captures/%s/%s'%(d,filename)
                image_data = tf.gfile.FastGFile(image, 'rb').read()

                try:
                    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0':image_data})
                    prediction = predictions[0]
                    prediction = prediction.tolist()
                    max_value = max(prediction)
                    max_index = prediction.index(max_value)
                    predicted_label = labels[max_index]
                    if predicted_label == d:
                        hits+=1
                    else:
                        misses+=1
                    print("Proabability = %f, Actual = %s, Prediction = %s, Hits = %s, Misses = %s"%(max_value,d,predicted_label,hits,misses))

                except Exception as e:
                    print("Error making prediction.")
                    print e
                    errors +=1

print("Errors = %d"%(errors))






        