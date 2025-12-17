"""
Main Entry Point for Automatic Differentiation Project
======================================================

This script provides a unified interface for:
- Training neural networks
- Making predictions
- Running analysis
- Viewing results
"""

import argparse
import sys
import os
from train import train_model, plot_training_curves, save_results, load_mnist
from predict import load_model, predict, predict_single, evaluate_predictions, save_model
from analysis import measure_backward_performance, scaling_analysis, compute_hessian_eigenvalues, plot_hessian_eigenvalues


def train_command(args):
    """Train a neural network model"""
    print("=" * 70)
    print("TRAINING NEURAL NETWORK")
    print("=" * 70)
    
    # Determine activation function
    activation = args.activation.lower()
    if activation not in ['relu', 'tanh']:
        print(f"Error: Invalid activation '{activation}'. Choose 'relu' or 'tanh'.")
        return
    
    # Determine dataset
    dataset = args.dataset.lower()
    if dataset not in ['mnist', 'fashion']:
        print(f"Error: Invalid dataset '{dataset}'. Choose 'mnist' or 'fashion'.")
        return
    
    # Train model
    results = train_model(
        activation=activation,
        dataset=dataset,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        use_cv=args.cv
    )
    
    # Plot training curves
    plot_training_curves(results, save_dir=args.plot_dir)
    
    # Save results
    results_file = f'results_{activation}_{dataset}.json'
    save_results(results, results_file)
    
    # Save model if requested
    if args.save_model:
        model_file = f'model_{activation}_{dataset}.pkl'
        save_model(results['model'], model_file)
        print(f"\nModel saved to {model_file}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best test accuracy: {results['best_test_acc']:.4f}")
    print(f"Final test accuracy: {results['test_accs'][-1]:.4f}")
    print(f"Results saved to: {results_file}")
    print(f"Plots saved to: {args.plot_dir}/")


def predict_command(args):
    """Make predictions using a trained model"""
    print("=" * 70)
    print("MAKING PREDICTIONS")
    print("=" * 70)
    
    # Load model
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        print("Train a model first using: python main.py train")
        return
    
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    
    # Load test data
    print("Loading test data...")
    _, X_test, _, y_test = load_mnist()
    
    # Make predictions
    print("Making predictions...")
    predictions, probabilities = predict(model, X_test, batch_size=args.batch_size)
    
    # Evaluate
    accuracy = evaluate_predictions(y_test, predictions)
    
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total samples: {len(y_test)}")
    print(f"Correct predictions: {int(accuracy * len(y_test))}")
    
    # Show some examples
    if args.show_examples > 0:
        print(f"\nFirst {args.show_examples} predictions:")
        for i in range(min(args.show_examples, len(y_test))):
            pred_class = predictions[i]
            true_class = y_test[i]
            confidence = probabilities[i][pred_class]
            status = "✓" if pred_class == true_class else "✗"
            print(f"  Sample {i+1}: Predicted={pred_class}, True={true_class}, "
                  f"Confidence={confidence:.3f} {status}")


def analyze_command(args):
    """Run performance analysis"""
    print("=" * 70)
    print("PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    if args.analysis_type == 'performance':
        # Measure backward pass performance
        from nn import Sequential, Linear, ReLU
        
        print("\nCreating test model...")
        model = Sequential(
            Linear(784, 128),
            ReLU(),
            Linear(128, 128),
            ReLU(),
            Linear(128, 10)
        )
        
        measure_backward_performance(model, input_size=784, batch_size=args.batch_size)
        
    elif args.analysis_type == 'scaling':
        # Scaling analysis
        scaling_analysis()
        
    elif args.analysis_type == 'hessian':
        # Hessian eigenvalue computation
        if not args.model:
            print("Error: --model required for Hessian analysis")
            return
        
        print(f"Loading model from {args.model}...")
        model = load_model(args.model)
        
        print("Loading sample data...")
        X_train, _, y_train, _ = load_mnist()
        
        # Use subset for efficiency
        n_samples = min(args.n_samples, len(X_train))
        X_sample = X_train[:n_samples]
        y_sample = y_train[:n_samples]
        
        print(f"Computing Hessian eigenvalues for {n_samples} samples...")
        eigenvals = compute_hessian_eigenvalues(
            model, 
            X_sample, 
            y_sample,
            max_params=args.max_params
        )
        
        # If we have both ReLU and Tanh models, compare
        if args.compare:
            print("\nComputing for comparison model...")
            # This would require loading a second model
            # For now, just show current results
            pass
        
        print(f"\nEigenvalue statistics:")
        print(f"  Largest: {eigenvals[0]:.2e}")
        print(f"  Smallest: {eigenvals[-1]:.2e}")
        print(f"  Condition number: {eigenvals[0] / (eigenvals[-1] + 1e-10):.2e}")
        
    else:
        print(f"Error: Unknown analysis type '{args.analysis_type}'")


def compare_command(args):
    """Compare ReLU vs Tanh activations"""
    print("=" * 70)
    print("COMPARING ACTIVATION FUNCTIONS")
    print("=" * 70)
    
    dataset = args.dataset.lower()
    
    # Train both models
    print("\nTraining ReLU model...")
    results_relu = train_model(
        activation='relu',
        dataset=dataset,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size
    )
    
    print("\nTraining Tanh model...")
    results_tanh = train_model(
        activation='tanh',
        dataset=dataset,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size
    )
    
    # Plot both
    plot_training_curves(results_relu, save_dir=args.plot_dir)
    plot_training_curves(results_tanh, save_dir=args.plot_dir)
    
    # Save results
    save_results(results_relu, f'results_relu_{dataset}.json')
    save_results(results_tanh, f'results_tanh_{dataset}.json')
    
    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Metric':<30} {'ReLU':<15} {'Tanh':<15}")
    print("-" * 70)
    print(f"{'Best Test Accuracy':<30} {results_relu['best_test_acc']:<15.4f} {results_tanh['best_test_acc']:<15.4f}")
    print(f"{'Final Test Accuracy':<30} {results_relu['test_accs'][-1]:<15.4f} {results_tanh['test_accs'][-1]:<15.4f}")
    print(f"{'Best Test Loss':<30} {min(results_relu['test_losses']):<15.4f} {min(results_tanh['test_losses']):<15.4f}")
    print(f"{'Final Test Loss':<30} {results_relu['test_losses'][-1]:<15.4f} {results_tanh['test_losses'][-1]:<15.4f}")
    print("=" * 70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Automatic Differentiation Engine - Neural Network Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a ReLU model
  python main.py train --activation relu --epochs 30

  # Train and save model
  python main.py train --activation tanh --save-model

  # Make predictions
  python main.py predict --model model_relu_mnist.pkl

  # Run performance analysis
  python main.py analyze --type performance

  # Compare ReLU vs Tanh
  python main.py compare --epochs 30
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a neural network')
    train_parser.add_argument('--activation', type=str, default='relu', 
                             choices=['relu', 'tanh'],
                             help='Activation function (default: relu)')
    train_parser.add_argument('--dataset', type=str, default='mnist',
                             choices=['mnist', 'fashion'],
                             help='Dataset to use (default: mnist)')
    train_parser.add_argument('--epochs', type=int, default=30,
                             help='Number of training epochs (default: 30)')
    train_parser.add_argument('--lr', type=float, default=0.01,
                             help='Initial learning rate (default: 0.01)')
    train_parser.add_argument('--batch-size', type=int, default=64,
                             help='Batch size (default: 64)')
    train_parser.add_argument('--cv', action='store_true',
                             help='Use cross-validation')
    train_parser.add_argument('--save-model', action='store_true',
                             help='Save trained model to file')
    train_parser.add_argument('--plot-dir', type=str, default='plots',
                             help='Directory to save plots (default: plots)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model', type=str, required=True,
                               help='Path to trained model file (.pkl)')
    predict_parser.add_argument('--batch-size', type=int, default=64,
                               help='Batch size for prediction (default: 64)')
    predict_parser.add_argument('--show-examples', type=int, default=10,
                               help='Number of example predictions to show (default: 10)')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run performance analysis')
    analyze_parser.add_argument('--type', type=str, dest='analysis_type',
                               choices=['performance', 'scaling', 'hessian'],
                               default='performance',
                               help='Type of analysis to run (default: performance)')
    analyze_parser.add_argument('--model', type=str,
                               help='Model file for Hessian analysis')
    analyze_parser.add_argument('--batch-size', type=int, default=64,
                               help='Batch size (default: 64)')
    analyze_parser.add_argument('--n-samples', type=int, default=100,
                               help='Number of samples for Hessian (default: 100)')
    analyze_parser.add_argument('--max-params', type=int, default=500,
                               help='Max parameters for Hessian (default: 500)')
    analyze_parser.add_argument('--compare', action='store_true',
                               help='Compare with another model')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare ReLU vs Tanh')
    compare_parser.add_argument('--dataset', type=str, default='mnist',
                               choices=['mnist', 'fashion'],
                               help='Dataset to use (default: mnist)')
    compare_parser.add_argument('--epochs', type=int, default=30,
                               help='Number of training epochs (default: 30)')
    compare_parser.add_argument('--lr', type=float, default=0.01,
                               help='Initial learning rate (default: 0.01)')
    compare_parser.add_argument('--batch-size', type=int, default=64,
                               help='Batch size (default: 64)')
    compare_parser.add_argument('--plot-dir', type=str, default='plots',
                               help='Directory to save plots (default: plots)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    try:
        if args.command == 'train':
            train_command(args)
        elif args.command == 'predict':
            predict_command(args)
        elif args.command == 'analyze':
            analyze_command(args)
        elif args.command == 'compare':
            compare_command(args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()



