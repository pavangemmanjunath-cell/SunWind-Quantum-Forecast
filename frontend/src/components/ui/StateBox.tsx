import type { ReactNode } from 'react';

interface Props { title: string; children?: ReactNode; error?: boolean }

export default function StateBox({ title, children, error }: Props) {
  return (
    <div className={`state-box ${error ? 'err' : ''}`} role={error ? 'alert' : 'status'}>
      <strong>{title}</strong>
      {children}
    </div>
  );
}
