import type { ButtonHTMLAttributes, ElementType } from 'react';
import { forwardRef } from 'react';
import { cn } from '../../lib/utils';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  asChild?: boolean;
  as?: ElementType;
}

function isAnchorElement(element: ElementType): boolean {
  return element === 'a';
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', loading, disabled, children, asChild, as, ...props }, ref) => {
    const baseStyles = 'inline-flex items-center justify-center gap-2 font-medium rounded-md transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-bg-primary disabled:opacity-50 disabled:cursor-not-allowed';

    const variants = {
      primary: 'bg-gradient-to-r from-accent-primary to-accent-tertiary text-text-inverse shadow-glow hover:shadow-glow-lg active:scale-[0.98]',
      secondary: 'bg-bg-tertiary text-text-primary border border-border-primary hover:bg-bg-hover hover:border-border-secondary',
      ghost: 'bg-transparent text-text-secondary hover:bg-bg-tertiary hover:text-text-primary',
      danger: 'bg-gradient-to-r from-error to-red-400 text-text-inverse shadow-[0_0_20px_rgba(255,71,87,0.3)] hover:shadow-[0_0_30px_rgba(255,71,87,0.5)]',
    };

    const sizes = {
      sm: 'px-3 py-1.5 text-xs',
      md: 'px-4 py-2 text-sm',
      lg: 'px-6 py-3 text-base',
    };

    const Comp = asChild ? (as || 'button') : 'button';
    const isAnchor = asChild && isAnchorElement(as || 'button');

    const buttonProps = {
      ref,
      className: cn(baseStyles, variants[variant], sizes[size], className),
      disabled: disabled || loading,
      ...(isAnchor ? {} : { type: 'button' }),
      ...props,
    } as any;

    // Remove asChild and as from props passed to DOM
    delete buttonProps.asChild;
    delete buttonProps.as;

    return (
      <Comp {...buttonProps}>
        {loading && (
          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="3"
              fill="none"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        )}
        {children}
      </Comp>
    );
  }
);

Button.displayName = 'Button';

export { Button };