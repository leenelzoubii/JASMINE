import { ProfessionalSidebar } from '@/components/layout/professional-sidebar';

export default function ProfessionalLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-dark-bg">
      <ProfessionalSidebar />
      <main className="lg:pl-64 min-h-screen">
        <div className="p-6 lg:p-8">{children}</div>
      </main>
    </div>
  );
}