-- Hyperparameters
local learningRate = 0.01
local numEpochs = 1000000

local function sigmoid(x)
    return 1 / (1 + math.exp(-x))
end

local function sigmoidDerivative(x)
    return sigmoid(x) * (1 - sigmoid(x))
end

local function createNeuralNetwork(inputSize, hiddenSize, outputSize)
    local neuralNetwork = {
        weightsInputHidden = {},
        weightsHiddenOutput = {},
        biasHidden = {},
        biasOutput = {},
		
        forward = function(self, input)
            local hiddenLayer = {}
            for i = 1, hiddenSize do
                local sum = 0
                for j = 1, inputSize do
                    sum = sum + input[j] * self.weightsInputHidden[j][i]
                end
                hiddenLayer[i] = sigmoid(sum + self.biasHidden[i])
            end

            local outputLayer = {}
            for i = 1, outputSize do
                local sum = 0
                for j = 1, hiddenSize do
                    sum = sum + hiddenLayer[j] * self.weightsHiddenOutput[j][i]
                end
                outputLayer[i] = sigmoid(sum + self.biasOutput[i])
            end

            return outputLayer
        end,
		
        backward = function(self, input, target)
            local hiddenLayer = {}
            for i = 1, hiddenSize do
                local sum = 0
                for j = 1, inputSize do
                    sum = sum + input[j] * self.weightsInputHidden[j][i]
                end
                hiddenLayer[i] = sigmoid(sum + self.biasHidden[i])
            end

            local outputLayer = {}
            for i = 1, outputSize do
                local sum = 0
                for j = 1, hiddenSize do
                    sum = sum + hiddenLayer[j] * self.weightsHiddenOutput[j][i]
                end
                outputLayer[i] = sigmoid(sum + self.biasOutput[i])
            end

            local outputDelta = {}
            for i = 1, outputSize do
                outputDelta[i] = (target[i] - outputLayer[i]) * sigmoidDerivative(outputLayer[i])
            end

            local hiddenDelta = {}
            for i = 1, hiddenSize do
                local sum = 0
                for j = 1, outputSize do
                    sum = sum + self.weightsHiddenOutput[i][j] * outputDelta[j]
                end
                hiddenDelta[i] = sum * sigmoidDerivative(hiddenLayer[i])
            end

            for i = 1, hiddenSize do
                for j = 1, outputSize do
                    self.weightsHiddenOutput[i][j] = self.weightsHiddenOutput[i][j] + learningRate * hiddenLayer[i] * outputDelta[j]
                end
            end

            for i = 1, inputSize do
                for j = 1, hiddenSize do
                    self.weightsInputHidden[i][j] = self.weightsInputHidden[i][j] + learningRate * input[i] * hiddenDelta[j]
                end
            end

            for i = 1, outputSize do
                self.biasOutput[i] = self.biasOutput[i] + learningRate * outputDelta[i]
            end

            for i = 1, hiddenSize do
                self.biasHidden[i] = self.biasHidden[i] + learningRate * hiddenDelta[i]
            end
        end
    }

    for i = 1, inputSize do
        neuralNetwork.weightsInputHidden[i] = {}
        for j = 1, hiddenSize do
            neuralNetwork.weightsInputHidden[i][j] = math.random()
        end
    end

    for i = 1, hiddenSize do
        neuralNetwork.weightsHiddenOutput[i] = {}
        for j = 1, outputSize do
            neuralNetwork.weightsHiddenOutput[i][j] = math.random()
        end
    end

    for i = 1, hiddenSize do
        neuralNetwork.biasHidden[i] = math.random()
    end

    for i = 1, outputSize do
        neuralNetwork.biasOutput[i] = math.random()
    end

    return neuralNetwork
end


-- Data Preparation
local imageSize = 28  -- Assuming input images are 28x28 pixels

local function getImageData(img)
    local out = {}
    local idx = 0
    for x=0, imageSize, 1 do
        for y=0, imageSize, 1 do
            -- get each pixel and remap it from 0-255 to 0-1
            out[idx] = math.Remap(img:GetColor(x, y).r, 0, 255, 0, 1) 
            idx = idx + 1
        end 
    end
    return out
end

-- Load Dataset
local function loadDataset(path)
    print("LOADING DATA SET: " .. path)
    local dataset = {}
    local max = 1000
    for i = 0, 9 do
        print("Loading files for " .. i)
        local digitPath = path .. i
        local x = 0
        for _, f in ipairs(file.Find("materials/" .. digitPath .. "/*.png", "GAME")) do
            local imagePath = digitPath .. "/" .. f
			msg1 = "Loading " .. imagePath
            
            local img = Material(imagePath)
            local label = i

            local imageTensor = getImageData(img)
            if x == 0 then
                PrintTable(imageTensor)
            end

            table.insert(dataset, {image = imageTensor, label = label})
            x = x + 1
            if x >= max then
                break
            end
			
			coroutine.yield()
        end
        print("..." .. x)
    end
    return dataset
end

-- Training
local function trainNeuralNetwork(neuralNetwork, dataset)
    for epoch = 1, numEpochs do
		msg1 = "Training " .. epoch .. "/" .. numEpochs
        for _, data in ipairs(dataset) do
            local input = data.image
            local target = {}
            for i = 1, 10 do
                target[i] = i == data.label and 1 or 0
            end

            -- Forward pass
            local prediction = neuralNetwork:forward(input)

            -- Backpropagation
            neuralNetwork:backward(input, target)
			coroutine.yield()
        end
    end
end

-- Testing
local function testNeuralNetwork(neuralNetwork, dataset)
    local correctPredictions = 0
	local idx = 1
	local mx = #dataset
    for _, data in ipairs(dataset) do
		msg1 = "Testing " .. idx .. "/" .. mx
		idx = idx + 1
		
        local input = getImageData(data.image)
        local target = data.label

        -- Forward pass
        local prediction = neuralNetwork:forward(input)

        -- Find the predicted label (class) with the highest probability
        local _, predictedLabel = torch.max(prediction, 1)
        predictedLabel = predictedLabel[1]

        if predictedLabel == target then
            correctPredictions = correctPredictions + 1
        end
		coroutine.yield()
    end

    local accuracy = correctPredictions / #dataset * 100
    print("Accuracy: " .. accuracy .. "%")
end

local msg1 = "Pending"
hook.Add("HUDPaint", "nn", function ()
    draw.WordBox(3, 5, 5, msg1, "DebugFixed", Color(0, 0, 0, 255), Color(255, 255, 255, 255))
end)

-- Load and preprocess the dataset
xalutils.ThinkCoroutine(function ()
		
	local training_data = loadDataset("mnist_png/training/")
	local testing_data = loadDataset("mnist_png/testing/")

	-- Neural Network Initialization
	local neuralNetwork = createNeuralNetwork(imageSize * imageSize, 64, 10)
	trainNeuralNetwork(neuralNetwork, training_data)
	return false
end)
