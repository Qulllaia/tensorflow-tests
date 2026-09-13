import * as tf from "@tensorflow/tfjs-node";

const loadedModel = await tf.loadLayersModel(
  "file://./models/my-linear-model/model.json",
);
const answer = loadedModel.predict(tf.tensor([5, 6, 7, 8, 9, 10]));
answer.print();
