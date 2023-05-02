using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Windows.AI.MachineLearning;
using Windows.Storage;

public class ClassifierInput
{
    public ImageFeatureValue Data { get; set; }
}

public class ClassifierOutput
{
    public TensorFloat Predictions { get; set; }

    public ClassifierOutput(TensorFloat outputPredictions)
    {
        Predictions = outputPredictions;
    }
}

public sealed class ClassifierModel
{
    private LearningModel _model;
    private LearningModelSession _session;

    public static async Task<ClassifierModel> CreateFromOnnxModelAsync(StorageFile file)
    {
        ClassifierModel model = new ClassifierModel();
        model._model = await LearningModel.LoadFromStorageFileAsync(file);
        model._session = new LearningModelSession(model._model);
        return model;
    }

    public async Task<ClassifierOutput> EvaluateAsync(ClassifierInput input)
    {
        LearningModelBinding binding = new LearningModelBinding(_session);
        binding.Bind("inputs", input.Data);
        LearningModelEvaluationResult result = await _session.EvaluateAsync(binding, string.Empty);
        var outputPredictions = result.Outputs["predictions"] as TensorFloat;
        return new ClassifierOutput(outputPredictions);
    }
    
    //foreach (var inputFeature in model._model.InputFeatures)
    //{
    //    System.Diagnostics.Debug.WriteLine($"Name: {inputFeature.Name}, Type: {inputFeature.Kind}, Shape: {inputFeature.Shape}");
    //}

}
