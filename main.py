from import_library import *
from data_processing import DataProcessing
from trainer import train_action_model, ActionModelTrainer
from younghee import YoungHee

def prepare_data():
    """
    Prepare the KTH dataset
    """
    print("Preparing data...")
    processor = DataProcessing()
    processor.process_kth_for_younghee()
    print("Preparing data complete!")
    return "data/processed/younghee/dataset.pkl"

def train_model(dataset_path, epochs, batch_size, learning_rate):
    """
    Train action recognition model
    """
    print(f"Training action recognition model with {epochs} epochs...")
    model_dir, accuracy = train_action_model(
        dataset_path, 
        batch_size=batch_size, 
        learning_rate=learning_rate, 
        epochs=epochs
    )
    print(f"Model training complete! Test accuracy: {accuracy:.4f}")
    return os.path.join(model_dir, "best_model_acc.pth")

def run_game(model_path, difficulty="easy"):
    """
    Run the Red Light Green Light game
    """
    print(f"Red Light Green Light game with difficulty: {difficulty}")
    game = YoungHee(model_path=model_path, difficulty=difficulty, num_classes=4)
    game.run()

def main():
    parser = argparse.ArgumentParser(description="Red Light Green Light Game with Action Recognition")
    parser.add_argument("--prepare", action="store_true", help="Prepare dataset")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--play", action="store_true", help="Run the game")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "hard"], 
                        help="Game difficulty")
    parser.add_argument("--dataset-path", type=str, default="data/processed/younghee/dataset.pkl", 
                        help="Path to processed dataset")
    parser.add_argument("--model-path", type=str, 
                        help="Path to trained model")
    
    args = parser.parse_args()
    
    if args.prepare:
        dataset_path = prepare_data()
        print(f"Dataset path: {dataset_path}")
    
    if args.train:
        if not os.path.exists(args.dataset_path):
            print(f"Dataset not found at {args.dataset_path}.")
            return
        
        model_path = train_model(
            args.dataset_path, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            learning_rate=args.learning_rate
        )
        print(f"Model trained and saved to: {model_path}")
    
    if args.play:
        if not os.path.exists(args.model_path):
            print(f"Model not found.")
            return
        
        run_game(args.model_path, args.difficulty)

if __name__ == "__main__":
    main()