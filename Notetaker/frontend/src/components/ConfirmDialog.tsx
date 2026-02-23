import { XMarkIcon } from '@heroicons/react/24/outline';

interface ConfirmDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  confirmButtonColor?: 'red' | 'blue' | 'green';
  isLoading?: boolean;
}

export const ConfirmDialog: React.FC<ConfirmDialogProps> = ({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmText = 'Sí, borrar',
  cancelText = 'Cancelar',
  confirmButtonColor = 'red',
  isLoading = false,
}) => {
  if (!isOpen) return null;

  const confirmButtonClasses = {
    red: 'bg-red-600 hover:bg-red-700 text-white',
    blue: 'bg-blue-600 hover:bg-blue-700 text-white',
    green: 'bg-green-600 hover:bg-green-700 text-white',
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen px-4">
        {/* Overlay */}
        <div
          className="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
          onClick={onClose}
        />

        {/* Modal */}
        <div className="relative bg-white dark:bg-slate-800 rounded-lg shadow-xl max-w-md w-full transition-colors">
          <div className="p-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-slate-50">
                {title}
              </h3>
              <button
                onClick={onClose}
                disabled={isLoading}
                className="text-gray-400 dark:text-slate-500 hover:text-gray-600 dark:hover:text-slate-300 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            </div>

            {/* Message */}
            <p className="text-gray-700 dark:text-slate-300 mb-6">
              {message}
            </p>

            {/* Buttons */}
            <div className="flex space-x-3">
              <button
                onClick={onConfirm}
                disabled={isLoading}
                className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${confirmButtonClasses[confirmButtonColor]}`}
              >
                {isLoading ? 'Procesando...' : confirmText}
              </button>
              <button
                onClick={onClose}
                disabled={isLoading}
                className="flex-1 px-4 py-2 bg-gray-200 dark:bg-slate-700 text-gray-800 dark:text-slate-200 rounded-lg font-medium hover:bg-gray-300 dark:hover:bg-slate-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {cancelText}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

