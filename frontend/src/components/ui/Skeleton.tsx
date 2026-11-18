import type { HTMLAttributes } from 'react';
import { forwardRef } from 'react';
import { cn } from '../../lib/utils';

export interface SkeletonProps extends HTMLAttributes<HTMLDivElement> {
  variant?: 'text' | 'circular' | 'rectangular';
  width?: string | number;
  height?: string | number;
  lines?: number;
}

export const Skeleton = forwardRef<HTMLDivElement, SkeletonProps>(
  ({ className, variant = 'text', width, height, lines = 1, ...props }, ref) => {
    const baseStyles = 'animate-shimmer rounded';

    const variants = {
      text: 'h-4 w-full',
      circular: 'rounded-full',
      rectangular: 'rounded-md',
    };

    if (variant === 'text' && lines > 1) {
      return (
        <div ref={ref} className={cn('space-y-2', className)} {...props}>
          {Array.from({ length: lines }).map((_, i) => (
            <div
              key={i}
              className={cn(baseStyles, variants.text, i === lines - 1 && 'w-3/4')}
              style={{ width: i === lines - 1 ? '75%' : '100%' }}
            />
          ))}
        </div>
      );
    }

    return (
      <div
        ref={ref}
        className={cn(baseStyles, variants[variant], className)}
        style={{ width, height, ...props.style }}
        {...props}
      />
    );
  }
);

Skeleton.displayName = 'Skeleton';

export interface SpinnerProps extends HTMLAttributes<HTMLDivElement> {
  size?: 'sm' | 'md' | 'lg';
}

export const Spinner = forwardRef<HTMLDivElement, SpinnerProps>(
  ({ className, size = 'md', ...props }, ref) => {
    const sizes = {
      sm: 'w-4 h-4 border-2',
      md: 'w-6 h-6 border-2',
      lg: 'w-10 h-10 border-3',
    };

    return (
      <div
        ref={ref}
        className={cn(
          'inline-block rounded-full border-current border-t-transparent animate-spin',
          sizes[size],
          className
        )}
        {...props}
        role="status"
        aria-label="Loading"
      >
        <span className="sr-only">Loading...</span>
      </div>
    );
  }
);

Spinner.displayName = 'Spinner';