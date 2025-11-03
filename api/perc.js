export default function handler(req, res) {
  res.send(`

    import numpy as np

def predict(xs, ws, b):
    xs = np.array(xs)
    net = np.dot(xs, ws) + b
    if net >= 0:
        pred = 1 
    else:
        pred = 0
    return pred

def train(xs, t, lr=0.1, no_e=10):
    x_size = xs.shape[1]
    
    ws = np.random.rand(x_size) * 0.1
    b = np.random.rand(1)[0] * 0.1
    print(f"Initial Parameters: Weights={ws}, Bias={b:.4f}")
    
    for e in range(no_e):
        flag = False
        for X, target in zip(xs, t):
            
            pred = predict(X, ws, b)
            
            error = target - pred
            
            if error != 0:
                flag = True
                ws += lr * error * X
                b += lr * error
        
        if not flag:
            break
            
    return ws, b

train_x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

and_y = np.array([0, 0, 0, 1])
or_y = np.array([0, 1, 1, 1])

print("Training for AND gate")
and_ws, and_b = train(train_x, and_y)
print("\nAND Gate training complete.\n")
print(f"Final AND Parameters: Weights={and_ws}, Bias={and_b:.4f}")

print("\nTraining for OR gate")
or_ws, or_b = train(train_x, or_y)
print("\nOR Gate training complete.\n")
print(f"Final OR Parameters: Weights={or_ws}, Bias={or_b:.4f} \n")


while True:
    x1 = int(input("Enter first input (0 or 1): "))
    x2 = int(input("Enter second input (0 or 1): "))
    if x1 in [0, 1] and x2 in [0, 1]:
        inputs=[x1, x2]
    else:
        print("Invalid input.Enter only 0 or 1.")
        continue
    and_result = predict(inputs, and_ws, and_b)
    print(f"Result: {inputs[0]} AND {inputs[1]} = {and_result}")

    or_result = predict(inputs, or_ws, or_b)
    print(f"Result: {inputs[0]} OR {inputs[1]} = {or_result}")
    break   
`);
}
