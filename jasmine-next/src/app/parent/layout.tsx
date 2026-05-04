import { ParentSidebar } from '@/components/layout/parent-sidebar';

export default function ParentLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-dark-bg">
      <ParentSidebar />
      <main className="lg:pl-64 min-h-screen">
        <div className="p-6 lg:p-8">{children}</div>
      </main>
    </div>
  );
}