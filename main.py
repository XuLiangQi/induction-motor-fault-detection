from app.tools.data_processing import pre_process_data
from app.build_model import build_model

if __name__ == "__main__":
    
    X_train, y_train, X_val, y_val, X_test, y_test = pre_process_data()

    build_model(X_train, y_train, X_val, y_val, X_test, y_test)