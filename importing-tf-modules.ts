import * as toxicity from "@tensorflow-models/toxicity";

const treshold = 0.5;

const classifier = await toxicity.load(treshold, ["toxicity"]);

const result = await classifier.classify("suck my balls");

console.log(JSON.stringify(result));
