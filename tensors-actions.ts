import * as tf from "@tensorflow/tfjs";
import * as tf_node from "@tensorflow/tfjs-node";
import { save3DBinaryArrayToPNG } from "./utils/image.generator.js";
import * as fs from "fs";
import * as path from "path";
import { pid } from "process";

// создать матрицу высотой 728, шириной 1024, полную нулей
const image1 = tf.zeros([728, 1024, 1]);

// то же самое, но с единицами
const image2 = tf.ones([728, 1024, 1]);

// создаст матрицу 200 на 200, заполненную 0.5 (серыми полупрозрачными клетками)
const image3 = tf.fill([200, 200, 4], 0.5);

// создаем маленький тайл
const smallTiles = tf.tensor([
  [[1], [0]],
  [[0], [1]],
]);

// увеличиваем его до размеров
const bigTile = smallTiles.tile([200, 200, 1]);
console.log(bigTile.rank);
console.log(bigTile.size);
console.log(bigTile.dtype);

// await save3DBinaryArrayToPNG(
//   bigTile.arraySync() as number[][][],
//   "bigTitle.png",
// );

// const shuffledImage = tf.randomUniform([200, 200, 1]);

// или: const shuffledImage = tf.randomUniform([200, 200, 1], 0, 255, "int32");
//
// await save3DBinaryArrayToPNG(
//   shuffledImage.arraySync() as number[][][],
//   "shuffledImage.png",
// );

const FILE_PATH = "files";
const cakeImagePath = path.join(FILE_PATH, "cake.png");
const cakeImage = fs.readFileSync(cakeImagePath);
let cakeBWTensor: tf.Tensor3D | tf.Tensor4D | undefined = undefined;

const tidyTensor = tf_node.tidy(() => {
  const cakeTensor = tf_node.node.decodeImage(cakeImage);
  console.log(`Success: local file to a ${cakeTensor.shape} tensor`);

  cakeBWTensor = tf_node.node.decodeImage(cakeImage, 1);

  console.log(`Success: local file to a ${cakeBWTensor.shape} tensor`);
  tf_node.keep(cakeBWTensor);
  return cakeBWTensor;
});

if (cakeBWTensor) {
  tf_node.node.encodePng(cakeBWTensor).then((file) => {
    fs.writeFileSync("./out/bw_cake.png", file);
    console.log("written");
  });

  const reversedCake = tf_node.reverse(cakeBWTensor, 1);

  tf_node.node.encodePng(reversedCake).then((file) => {
    fs.writeFileSync("./out/reversed_cake.png", file);
    console.log("written");
  });

  const flipCake = tf_node.tidy(() => {
    const lol = tf_node.expandDims(tidyTensor, 0); // [1, H, W, 1]
    const flipped = tf_node.image.flipLeftRight(lol.asType("float32"));
    return tf_node.squeeze(flipped, [0]).asType("int32"); // [H, W, 1]
  });

  tf_node.node.encodePng(flipCake as tf.Tensor3D).then((file) => {
    fs.writeFileSync("./out/flipCake.png", file);
    console.log("written");
  });
}
