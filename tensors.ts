import * as tf from "@tensorflow/tfjs";

const dataArray = [1, 2, 3, 4, 5, 6];

const tensor = tf.tensor(dataArray); // или tf.tensor1d(dataArray);
// The second method provides an extra level of runtime checking since
// you’ve defined the expected dimensionality. Determining the desired
// dimensionality is useful when you’re looking to ensure the number of
// dimensions in the data you’re working with. Methods exist for verifying up
// to six dimensions with tf.tensor6d.

// чтобы tf.tensor(dataArray); - по дефолту создаст тензор float32
const intTensor = tf.tensor(dataArray, undefined, "int32");
// вы пример, как создать int32 тип
//

console.log(
  JSON.stringify({
    rank: tensor.rank,
    size: tensor.size,
    type: tensor.dtype,
  }),
);
// {"rank":1,"size":6,"type":"float32"}

// тензор из крестики-нолики

/*
 * ситуацию на доске:
 * x - -
 * - o -
 * x - -
 *
 * можно представить в виде одномерного массива
 *
 * [-1 0 0 0 1 0 -1 0 0] - где -1 = x, а 1 = 0
 * таким образом, получается, что у нас одномерный массив описывает ситуацию на доске
 * можно представить и как двумерный массив: [[-1 0 0], [0 1 0], [-1 0 0]]
 * */

// одномерный тензор
const ticTacToeTensor1d = tf.tensor([-1, 0, 0, 0, 1, 0, -1, 0, 0]);

// двумерный тензор
const ticTacToeTensor2d = tf.tensor([
  [-1, 0, 0],
  [0, 1, 0],
  [-1, 0, 0],
]);

// создание двумерного тензора с помощью параметра shape
const ticTacToeTensor2dViaShapParameter = tf.tensor(
  [-1, 0, 0, 0, 1, 0, -1, 0, 0],
  [3, 3],
);

// tip:
const nope = tf.tensor([4], undefined, "float32");
const yep = nope.asType("int32");
// yep - новый массив со значениями тензора типа int32, округленными на подобии Math.floor

// сборщик мусора не работает автоматически на тензоры, потому их придется отчищать вручную

console.log(tf.memory().numBytes);
console.log(tf.memory().numTensors);

let tidyKeep, tidyBeater, tidyReturn;

tf.tidy(() => {
  tidyKeep = tf.tensor([1, 2, 3]);

  tidyBeater = tf.tensor([1, 2, 3]);

  tidyReturn = tf.tensor([1, 2, 3]);

  console.log("inside tidy", tf.memory().numTensors);

  tf.keep(tidyKeep);
  return tidyReturn;
});

console.log("after tidy", tf.memory().numTensors);

tf.dispose(tidyKeep);
tf.dispose(tidyReturn);

console.log("finaly dispose", tf.memory().numTensors);
/*
(до этого кода считаются в том числе не отчищенные из памяти тензоры)
7
inside tidy 10
after tidy 9
finaly dispose 7
 * */

const tensorData = tf.tensor([
  [1, 2, 3],
  [4, 5, 6],
]);

// показывает структуру объекта тензора
console.log(tensorData);
// показывает содержимые данные тензора
console.log(tensorData.print());
// формирует массив из тензора
console.log(tensorData.arraySync());
// фактически выравнивает данные, формирует ответ из 2d в 1d
console.log(tensorData.dataSync());
// отчищаем
tf.dispose(tensorData);

// действия с тензорами

const math1 = [
  [1, 2, 3],
  [3, 4, 5],
  [6, 7, 8],
];

const math2 = [
  [1, 2, 3],
  [3, 4, 5],
  [6, 7, 8],
];

console.log(tf.matMul(math1, math2).print());

// практический пример того, как можно обрабатывать данные с помощью тензоров

const users = ["Gant", "Todd", "Jed", "Justin"];
const bands = [
  "Nirvana",
  "Nine Inch Nails",
  "Backstreet Boys",
  "N Sync",
  "Night Club",
  "Apashe",
  "STP",
];
const features = [
  "Grunge",
  "Rock",
  "Industrial",
  "Boy Band",
  "Dance",
  "Techno",
];
// User votes - оценки пользователей по каждой группе
const user_votes = tf.tensor([
  [10, 9, 1, 1, 8, 7, 8],
  [6, 8, 2, 2, 0, 10, 0],
  [0, 2, 10, 9, 3, 7, 0],
  [7, 4, 2, 3, 6, 5, 5],
]);
// Music Styles - соответствие жанрам каждой группы (группа - строка, жанр - столбец)
const band_feats = tf.tensor([
  [1, 1, 0, 0, 0, 0],
  [1, 0, 1, 0, 0, 0],
  [0, 0, 0, 1, 1, 0],
  [0, 0, 0, 1, 0, 0],
  [0, 0, 1, 0, 0, 1],
  [0, 0, 1, 0, 0, 1],
  [1, 1, 0, 0, 0, 0],
]);

// User's favorite styles - умножением матрицы пользовательских оценок на соответствие
// жанрам мы получаем пересечение предпочтений пользователей по жанрам
const user_feats = tf.matMul(user_votes, band_feats);
// Print the answers
user_feats.print();

const top_user_features = tf.topk(user_feats, features.length);
// Back to JavaScript
const top_genres = top_user_features.indices.arraySync();
// print the results
users.map((u, i) => {
  const rankedCategories = top_genres[i].map((v) => features[v]);
  console.log(u, rankedCategories);
});
