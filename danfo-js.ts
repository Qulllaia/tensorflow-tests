import * as dfd from "danfojs-node";

const df = await dfd.readCSV("./files/train.csv");
console.log("Train Size", df.shape[0]);

// голова дата фрейма
// df.head().print();

// описание данных, null заполняются 0
// df.fillNa(0).describe().print();

// const empty_spots = df.isNa().sum();
// empty_spots.print();
// Find the average
// const empty_rate = empty_spots.div(df.isNa().count());
// empty_rate.print();

const dft = await dfd.readCSV("./files/test.csv");
console.log("Test Size", dft.shape[0]);

const combinedDf = dfd.concat({ dfList: [df, dft], axis: 0 }) as dfd.DataFrame;
console.log("combinedDf Size", combinedDf.shape[0]);

combinedDf.fillNa(0).describe().print();

const cleanedDf = combinedDf.drop({
  columns: ["Name", "PassengerId", "Ticket", "Cabin"],
});

const cleanedDfWithoutNa = cleanedDf.dropNa().resetIndex();
console.log(
  `cleanedDfWithoutNa after cleaning: ${cleanedDfWithoutNa.shape[0]}`,
);
cleanedDfWithoutNa.head().print();

function encodeValue(dataFrame: dfd.DataFrame, columnName: string) {
  const encoder = new dfd.LabelEncoder();
  encoder.fit(dataFrame[columnName]);
  dataFrame[columnName] = encoder.transform(dataFrame[columnName].values);
}

encodeValue(cleanedDfWithoutNa, "Embarked");
encodeValue(cleanedDfWithoutNa, "Sex");
cleanedDfWithoutNa.head().print();

// создаем датасеты для будущей тренировки

const newTrain = await cleanedDfWithoutNa.sample(800);
console.log(`newTrain samle ${newTrain.shape[0]}`);

const newTest = cleanedDfWithoutNa.drop({ index: newTrain.index });
console.log(`newTest samle ${newTest.shape[0]}`);

dfd.toCSV(newTrain, {
  filePath: "./out/cleaned_csv/newTrain.csv",
});
dfd.toCSV(newTest, {
  filePath: "./out/cleaned_csv/newTest.csv",
});
console.log("Files written!");
