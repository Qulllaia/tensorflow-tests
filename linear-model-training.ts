import * as tf from "@tensorflow/tfjs-node";

const x_values = tf.tensor1d([-1, 0, 1, 2, 3, 4]);

const y_values = tf.tensor1d([-4, -2, 0, 2, 4, 6]);

const model = tf.sequential();

model.add(
  tf.layers.dense({
    inputDim: 1,
    units: 1,
  }),
);

model.compile({
  optimizer: "sgd",
  loss: "meanSquaredError",
});

model.summary();

await model.fit(x_values, y_values, { epochs: 1000 }).then((history) => {
  const inputTensor = tf.tensor([5, 6, 7, 8, 9, 10]);

  const answer = model.predict(inputTensor);
  console.log(`10 results in ${(answer as tf.Tensor<tf.Rank>).dataSync()}`);

  tf.dispose([answer, inputTensor]);
});

await model.save("file://./models/my-linear-model");

tf.dispose([x_values, y_values]);

const loadedModel = await tf.loadLayersModel(
  "file://./models/my-linear-model/model.json",
);
const answer = loadedModel.predict(tf.tensor([5, 6, 7, 8, 9, 10]));

if (!Array.isArray(answer)) {
  answer.print();
} else console.log("wtf");
