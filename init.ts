import * as dfd from "danfojs-node";
import * as tf from "@tensorflow/tfjs-node";

// Get cleaned data
const df = await dfd.readCSV("./out/cleaned_csv/newTrain.csv");
console.log("Train Size", df.shape[0]);
df.head().print();
const dft = await dfd.readCSV("./out/cleaned_csv/newTest.csv");
console.log("Test Size", dft.shape[0]);
// Split train into X/Y
const trainX = df.iloc({ columns: [`1:`] }).tensor as tf.Tensor;
const trainY = df["Survived"].tensor;

console.log(trainX.print());

// Split test into X/Y
const testX = dft.iloc({ columns: [`1:`] }).tensor;
const testY = dft["Survived"].tensor;

const model = tf.sequential();

const inputShape = [(df.shape[1] || 0) - 1];
model.add(
  tf.layers.dense({
    inputShape,
    units: 120,
    // Each layer is utilizing ReLU activation up until the final layer.
    activation: "relu",
    // This line tells the model to initialize weights based on an algorithm
    // rather than simply setting the model’s initial weights to complete
    // randomness. This sometimes helps a model start much closer to the
    // answer. It’s not critical in this case, but it’s a useful feature of
    // TensorFlow.js.
    kernelInitializer: "heNormal",
  }),
);
model.add(tf.layers.dense({ units: 64, activation: "relu" }));
model.add(tf.layers.dense({ units: 32, activation: "relu" }));
model.add(
  tf.layers.dense({
    units: 1,
    //The final layer uses sigmoid activation to print a number between zero and one (survived or did not survive).
    activation: "sigmoid",
  }),
);

model.summary();

model.compile({
  optimizer: "adam",
  loss: "binaryCrossentropy",
  metrics: ["accuracy"],
});

await model.fit(trainX, trainY, {
  batchSize: 32,
  epochs: 100,
  // данные, на которых мы будем валидироваться
  validationData: [testX, testY],
});

model.save("file://./models/titanic-predict/");

const loadedModel = await tf.loadLayersModel(
  "file://./models/titanic-predict/model.json",
);
const answer = loadedModel.predict(tf.tensor2d([[2, 1, 28, 1, 0, 26, 0]]));

if (!Array.isArray(answer)) {
  answer.print();
} else console.log("wtf");
