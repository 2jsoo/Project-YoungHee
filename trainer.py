from import_library import *

class ActionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=4, dropout=0.3):
        super(ActionClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Tanh()
        )
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self._init_weights()
    
    def forward(self, x):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Bi-LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification layers
        out = self.dropout(context)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


class ActionModelTrainer:
    def __init__(self, dataset_path, model_save_dir="models/action_classifier"):
        self.dataset_path = dataset_path
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Load dataset
        self.dataset = self.load_dataset(dataset_path)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def load_dataset(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        print(f"Dataset loaded: X_train shape = {dataset['X_train'].shape}")
        if 'class_names' in dataset:
            print(f"Classes: {dataset['class_names']}")
        return dataset
    
    def create_dataloaders(self, batch_size=32):
        X_train = torch.FloatTensor(self.dataset['X_train'])
        y_train = torch.LongTensor(self.dataset['y_train'])
        X_val = torch.FloatTensor(self.dataset['X_val'])
        y_val = torch.LongTensor(self.dataset['y_val'])
        X_test = torch.FloatTensor(self.dataset['X_test'])
        y_test = torch.LongTensor(self.dataset['y_test'])
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, batch_size, learning_rate, epochs):
        num_classes = len(self.dataset['class_names']) # ['boxing', 'handclapping', 'handwaving', 'walking']
        
        # ActionClassifier hyperparameters
        hidden_size = 128
        num_layers = 2
        dropout = 0.3
        
        # Save path
        model_name = f"action_classifier_batch{batch_size}_lr{learning_rate}_epochs{epochs}"
        save_dir = os.path.join(self.model_save_dir, model_name)
        os.makedirs(save_dir, exist_ok=True)
        
        input_size = self.dataset['X_train'].shape[-1]  # Features per frame
        model = ActionClassifier(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            num_classes=num_classes,
            dropout=dropout
        ).to(self.device)
        
        train_loader, val_loader, _ = self.create_dataloaders(batch_size)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
        
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_loss, best_val_acc = float('inf'), 0.0
        
        # Training Action Classifier
        for epoch in tqdm(range(epochs), desc="Training Action Classifier"):
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == targets).sum().item()
                train_total += targets.size(0)
            
            train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == targets).sum().item()
                    val_total += targets.size(0)
            
            val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            val_losses.append(val_loss)
            val_accs.append(val_accuracy)
            
            scheduler.step(val_loss)
            
            # Check progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Save best model (minimun val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'class_names': self.dataset.get('class_names', ['boxing', 'handclapping', 'handwaving', 'walking']),
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout
                }, os.path.join(save_dir, "best_model_loss.pth"))
            
            # Save best model by validation accuracy
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_accuracy,
                    'class_names': self.dataset.get('class_names', ['boxing', 'handclapping', 'handwaving', 'walking']),
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout
                }, os.path.join(save_dir, "best_model_acc.pth"))
        
        # Save final model
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1],
            'val_acc': val_accs[-1],
            'class_names': self.dataset.get('class_names', ['boxing', 'handclapping', 'handwaving', 'walking']),
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout
        }, os.path.join(save_dir, "final_model.pth"))
        
        # Plot training curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train')
        plt.plot(val_accs, label='Validation')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_curves.png"))
        
        return model, save_dir
    
    def evaluate_model(self, model_path=None, batch_size=32):
        """Evaluate the model"""
        class_names = self.dataset.get('class_names', ['boxing', 'handclapping', 'handwaving', 'walking'])
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Hyperparameters of model
        hidden_size = checkpoint.get('hidden_size', 128)
        num_layers = checkpoint.get('num_layers', 2)
        dropout = checkpoint.get('dropout', 0.3)
        
        input_size = self.dataset['X_train'].shape[-1]
        num_classes = len(class_names)
        
        model = ActionClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
        
        # Create test dataloader
        _, _, test_loader = self.create_dataloaders(batch_size)
        
        # Evaluate model
        model.eval()
        test_correct, test_total = 0, 0
        all_predictions, all_targets = [], []
        class_correct = [0] * len(class_names)
        class_total = [0] * len(class_names)
        
        confusion_mat = np.zeros((len(class_names), len(class_names)), dtype=int)
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Evaluating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = model(inputs)
                
                _, predicted = torch.max(outputs.data, 1)
                test_correct += (predicted == targets).sum().item()
                test_total += targets.size(0)
                
                # Class-wise accuracy
                for i in range(targets.size(0)):
                    label = targets[i].item()
                    pred = predicted[i].item()
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
                    confusion_mat[label, pred] += 1
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_accuracy = test_correct / test_total
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        print("\nClass-wise Accuracy:")
        for i in range(len(class_names)):
            if class_total[i] > 0:
                accuracy = class_correct[i] / class_total[i]
                print(f"  {class_names[i]}: {accuracy:.4f} ({class_correct[i]}/{class_total[i]})")
        
        print("\nConfusion Matrix:")
        print(confusion_mat)
        
        # Visualize confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        thresh = confusion_mat.max() / 2.
        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                plt.text(j, i, format(confusion_mat[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if confusion_mat[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save results
        save_dir = os.path.dirname(model_path)
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
        plt.close()
        
        return test_accuracy, confusion_mat


def train_action_model(dataset_path, batch_size=32, learning_rate=1e-3, epochs=30):
    """
    Train action classification model
    """
    trainer = ActionModelTrainer(dataset_path)
    model, save_dir = trainer.train_model(batch_size, learning_rate, epochs)
    test_accuracy, _ = trainer.evaluate_model(os.path.join(save_dir, "best_model_acc.pth"))
    return save_dir, test_accuracy