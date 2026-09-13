import * as tf from "@tensorflow/tfjs-node";

const prepareData = () => {
  const tempX = [];
  const tempY = [];

  const step = 0.0001;
  const dataSize = 10;

  for (let i = 0; i < dataSize; i += step) {
    tempX.push(i);
    tempY.push(i * i);
  }
  return [tempX, tempY];
};

const printCallback = {
  onEpochEnd: (epoch: number, log: any) => {
    console.log(epoch, log);
  },
};

async function main() {
  const [x_values, y_values] = prepareData();

  if (!x_values || !y_values) return;

  const x_tensors = tf.tensor(x_values);
  const y_tensors = tf.tensor(y_values);

  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      units: 20,
      inputDim: 1,
      activation: "relu",
    }),
  );

  model.add(
    tf.layers.dense({
      units: 20,
      activation: "relu",
    }),
  );

  model.add(
    tf.layers.dense({
      units: 1,
    }),
  );

  model.compile({
    optimizer: "adam",
    loss: "meanSquaredError",
  });

  model.summary();

  await model.fit(x_tensors, y_tensors, {
    epochs: 30,
    callbacks: printCallback,
    batchSize: 64,
  });

  await model.save("file://./models/my-non-linear-model/");
  const answer = model.predict(tf.tensor([7]));
  if (Array.isArray(answer)) {
    console.log("wtf");
    return;
  }
  answer.print();
  answer.dispose();
  x_tensors.dispose();
  y_tensors.dispose();
  model.dispose();
}

main();
