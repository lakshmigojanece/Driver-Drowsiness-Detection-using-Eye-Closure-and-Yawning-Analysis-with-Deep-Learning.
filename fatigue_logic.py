def fatigue_stage(prediction):

    if prediction == "open" or prediction == "no_yawn":
        return "Alert",0

    elif prediction == "yawn":
        return "Mild Fatigue",1

    elif prediction == "closed":
        return "Severe Fatigue",2


# example
pred = "yawn"

stage,level = fatigue_stage(pred)

print("Fatigue Stage:",stage)