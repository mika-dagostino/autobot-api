from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch.nn.functional as F
import numpy as np

epsilon1 = 0.075
epsilon2 = 0.3

ref_dict = {"MAKE":"Make", "MODEL":"Model", "BODY":"Body", "EXTERIOR COLOR":"ExteriorColor", "INTERIOR COLOR":"InteriorColor", "TRANSMISSION":"Transmission", "MARKET CLASS":"MarketClass", "DRIVETRAIN":"Drivetrain", "ENGINE DESCRIPTION":"Engine_Description", "DOORS":"Doors", "ENGINE CYLINDERS":"EngineCylinders", "MILES":"Miles", "SELLING PRICE":"SellingPrice", "PASSENGER CAPACITY":"PassengerCapacity", "CITYMPG":"CityMPG", "HIGHWAYMPG":"HighwayMPG", "YEAR":"Year"}

model_name = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

def absaList(phrase, aspects):
  outputDict = {}

  for TAG in aspects.keys():
    if TAG in ["MAKE", "MODEL", "BODY", "EXTERIOR COLOR", "INTERIOR COLOR", "TRANSMISSION", "MARKET CLASS", "DRIVETRAIN", "ENGINE DESCRIPTION"]:
      process_prepends(aspects, TAG, phrase, outputDict, ["hate", "okay", "love"])
    elif TAG == "YEAR":
      process_prepends(aspects, TAG, phrase, outputDict, ["before", "", "after"])
    else:
      process_prepends(aspects, TAG, phrase, outputDict, ["less", "", "more"])

  return outputDict


# probs = (negative, neutral, positive)
# prepends = (negative, neutral, positive) or (before, 'none', after) or (less, 'none', more)
def assertSentiment(probs, prepends):
  if ((np.max(probs) - np.min(probs)) < epsilon1) or (abs(probs[1] - probs[0]) > epsilon2 and abs(probs[1] - probs[2]) > epsilon2):
    return prepends[1]
  elif probs[0] > probs[2]:
    return prepends[0]
  else:
    return prepends[2]


def queryModel(token, phrase):
  inputs = tokenizer(f"[CLS] {phrase} [SEP] {token} [SEP]", return_tensors="pt")
  outputs = model(**inputs)
  probs = F.softmax(outputs.logits, dim=1)
  probs = probs.detach().numpy()[0]

  return probs


def process_prepends(aspects, TAG, phrase, outputDict, prepends):
  if type(aspects[TAG]) is list:
    for val in aspects[TAG]:
      probs = queryModel(val, phrase)
      stringToAdd = assertSentiment(probs, prepends) + " "
      stringToAdd += val

      if not ref_dict[TAG] in outputDict:
        outputDict.update({ref_dict[TAG]: stringToAdd})
      else:
        temp = outputDict[ref_dict[TAG]]
        if type(temp) is tuple:
          outputDict.update({ref_dict[TAG]: temp + (stringToAdd,)})
        else:
          outputDict.update({ref_dict[TAG]: (temp,) + (stringToAdd,)})

  else:
    probs = queryModel(aspects[TAG], phrase)
    stringToAdd = assertSentiment(probs, prepends) + " "

    stringToAdd += aspects[TAG]
    outputDict.update({ref_dict[TAG]: stringToAdd})